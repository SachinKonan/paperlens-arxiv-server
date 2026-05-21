"""PaperLens vision reranker: score(papers) -> list[p_accept].

Mirrors the lab's vision inference pipeline (RANKER.md §4.2 + scripts/vllm_infer.py
in paperlens-training-and-inference):

  1. Build ShareGPT messages [system, user(prompt + title + abstract + N <image>)].
  2. Apply qwen2_vl chat template with add_generation_prompt=True so the
     assistant turn starts fresh.
  3. Pass per-paper PIL page images via vLLM's multi_modal_data={"image": [...]}.
  4. Greedy sample max_tokens=8 with logprobs=5 -- captures the top-5
     token logprobs at every step.
  5. The decision token is at position DECISION_TOKEN_IDX=5 in the
     generated sequence: Outcome / : / " \\" / boxed / { / Accept|Reject.
  6. Score per RANKER.md §4.2:
        p_accept = exp(logp_accept[5]) / (exp(logp_accept[5]) + exp(logp_reject[5]))
     This is the 2-token softmax (calibrated to {Accept, Reject}), in [0, 1].

Identical to what RANKER.md §6.1 ran offline to produce predictions_3b.parquet,
which the cache layer can bootstrap from.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from .prompts import SHAREGPT_SYSTEM_PROMPT, build_user_turn
from .retriever_client import RetrievedPaper


log = logging.getLogger(__name__)

# Position of the Accept|Reject token in "Outcome: \\boxed{X" under the
# Qwen2.5 BPE tokenizer (5 leading tokens: Outcome / : / " \\" / boxed / { ).
DECISION_TOKEN_IDX = 5
ACCEPT_TOKEN = "Accept"
REJECT_TOKEN = "Reject"


def _softmax2(logp_a: float, logp_b: float) -> float:
    """Renormalized 2-token softmax. Returns p(a)/(p(a)+p(b))."""
    # Stable form: subtract max from both before exp
    m = max(logp_a, logp_b)
    ea = math.exp(logp_a - m)
    eb = math.exp(logp_b - m)
    return ea / (ea + eb)


class PaperLensReranker:
    def __init__(
        self,
        ckpt_path: str,
        *,
        modality: str = "vision",
        domain: str = "arxiv",
        template: str = "qwen2_vl",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 24576,
    ):
        self.ckpt_path = ckpt_path
        self.modality = modality.lower()
        self.domain = domain.lower()
        self.template = template
        if self.modality not in ("text", "vision"):
            raise ValueError(f"modality must be text|vision, got {modality!r}")
        if self.domain not in ("arxiv", "iclr"):
            raise ValueError(f"domain must be arxiv|iclr, got {domain!r}")

        log.info(
            f"loading PaperLens reranker: ckpt={ckpt_path} modality={modality} "
            f"domain={domain} template={template}"
        )
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        llm_kwargs = dict(
            model=ckpt_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        if self.modality == "vision":
            # Cap per-prompt images conservatively; per_venue papers have ~7-14 pages.
            llm_kwargs["limit_mm_per_prompt"] = {"image": 20}
        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            logprobs=5,
        )

        accept_ids = self.tokenizer.encode(ACCEPT_TOKEN, add_special_tokens=False)
        reject_ids = self.tokenizer.encode(REJECT_TOKEN, add_special_tokens=False)
        assert len(accept_ids) == 1, f"'Accept' tokenizes to {len(accept_ids)}: {accept_ids!r}"
        assert len(reject_ids) == 1, f"'Reject' tokenizes to {len(reject_ids)}: {reject_ids!r}"
        self.accept_id = accept_ids[0]
        self.reject_id = reject_ids[0]
        log.info(f"decision tokens: Accept={self.accept_id}, Reject={self.reject_id}")

    def _build_messages(self, paper: RetrievedPaper, n_images: int) -> list[dict]:
        body = paper.full_text or ""
        user_value = build_user_turn(
            domain=self.domain,
            title=paper.title,
            body=body,
            n_images=n_images,
            abstract=paper.abstract,
        )
        return [
            {"role": "system", "content": SHAREGPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_value},
        ]

    def _render_prompt(self, paper: RetrievedPaper, n_images: int) -> str:
        messages = self._build_messages(paper, n_images)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def score(self, papers: list[RetrievedPaper]) -> list[float]:
        """Return p_accept in [0, 1] per paper.

        For vision papers, each ``paper.images`` must already be populated
        with PIL.Image objects (call ``image_loader.load_pages(arxiv_id)``
        upstream and attach them).
        """
        if not papers:
            return []

        vllm_inputs = []
        for p in papers:
            n_images = len(p.images) if p.images else 0
            prompt = self._render_prompt(p, n_images)
            req = {"prompt": prompt}
            if self.modality == "vision" and n_images > 0:
                req["multi_modal_data"] = {"image": list(p.images)}
            vllm_inputs.append(req)

        outputs = self.llm.generate(vllm_inputs, self.sampling_params)

        scores: list[float] = []
        for out in outputs:
            per_step = out.outputs[0].logprobs or []
            if len(per_step) <= DECISION_TOKEN_IDX:
                log.warning(
                    f"generation too short ({len(per_step)} steps) for decision; "
                    f"defaulting p_accept=0.5"
                )
                scores.append(0.5)
                continue
            step_lp = per_step[DECISION_TOKEN_IDX] or {}
            la_o = step_lp.get(self.accept_id)
            lr_o = step_lp.get(self.reject_id)
            la = float(la_o.logprob) if hasattr(la_o, "logprob") else (float(la_o) if la_o is not None else -50.0)
            lr = float(lr_o.logprob) if hasattr(lr_o, "logprob") else (float(lr_o) if lr_o is not None else -50.0)
            scores.append(_softmax2(la, lr))
        return scores

    def close(self) -> None:
        self.llm = None
