"""PaperLens reranker: score(papers, domain) -> list[float].

Loads a PaperLens HF model via vLLM. For each paper, builds the canonical
ShareGPT prompt (system + user_with_paper), then computes
``score = logprob(Accept) - logprob(Reject)`` at the boxed-token position
("Outcome: \\boxed{" + scored token).

Higher score = more likely to be accepted at a top venue.
"""
from __future__ import annotations

import logging
from typing import Optional

from .prompts import SHAREGPT_SYSTEM_PROMPT, build_user_turn
from .retriever_client import RetrievedPaper


log = logging.getLogger(__name__)

# We score the *first* sub-token of the literal "Accept" / "Reject" strings
# at the position immediately after the boxed-prefix. The base Qwen2.5
# tokenizer splits these as single tokens — but we look them up dynamically
# at init time so subword splits would still work.
_BOXED_PREFIX = "Outcome: \\boxed{"
ACCEPT_TOKEN = "Accept"
REJECT_TOKEN = "Reject"


class PaperLensReranker:
    def __init__(self, hf_repo: str, *, modality: str = "text", domain: str = "arxiv",
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.6,
                 max_model_len: int = 24576):
        self.hf_repo = hf_repo
        self.modality = modality.lower()
        self.domain = domain.lower()
        if self.modality not in ("text", "vision"):
            raise ValueError(f"modality must be text|vision, got {modality!r}")
        if self.domain not in ("arxiv", "iclr"):
            raise ValueError(f"domain must be arxiv|iclr, got {domain!r}")

        log.info(f"loading PaperLens reranker: hf_repo={hf_repo} modality={modality} domain={domain}")
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
        self.llm = LLM(
            model=hf_repo,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        # Greedy decode of 1 token; we read its logprob distribution.
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=20,
        )

        # Tokenize Accept / Reject as standalone strings to find their token IDs
        accept_ids = self.tokenizer.encode(ACCEPT_TOKEN, add_special_tokens=False)
        reject_ids = self.tokenizer.encode(REJECT_TOKEN, add_special_tokens=False)
        if not accept_ids or not reject_ids:
            raise RuntimeError("tokenizer returned empty ids for Accept/Reject")
        self.accept_id = accept_ids[0]
        self.reject_id = reject_ids[0]
        log.info(f"scoring token ids: Accept={self.accept_id}, Reject={self.reject_id}")

    def _build_messages(self, paper: RetrievedPaper) -> list[dict]:
        """Compose the ShareGPT-style messages list for one paper.

        We prefix the assistant turn with `Outcome: \\boxed{` so that the
        next-token logprobs slot the score across Accept / Reject.
        """
        body = paper.full_text or f"# {paper.title}\n\n## Abstract\n{paper.abstract}"
        user_value = build_user_turn(
            domain=self.domain,
            title=paper.title,
            body=body,
            n_images=0,                        # text reranker only (vision: TODO)
            abstract=paper.abstract,
        )
        return [
            {"role": "system", "content": SHAREGPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_value},
            {"role": "assistant", "content": _BOXED_PREFIX},
        ]

    def _render_prompt(self, paper: RetrievedPaper) -> str:
        """Apply the chat template, ending just before Accept|Reject."""
        messages = self._build_messages(paper)
        # `add_generation_prompt=False` because we've already supplied the
        # assistant turn (with the boxed prefix). The chat template will
        # include it as a continuation.
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # Some chat templates suffix an EOS token after the assistant turn;
        # strip if present so the model continues from `\boxed{` not after eos.
        if rendered.endswith(self.tokenizer.eos_token or "<|im_end|>"):
            rendered = rendered.rsplit(self.tokenizer.eos_token or "<|im_end|>", 1)[0]
        return rendered.rstrip()

    def score(self, papers: list[RetrievedPaper]) -> list[float]:
        """Return ``logprob(Accept) - logprob(Reject)`` for each paper."""
        if not papers:
            return []
        prompts = [self._render_prompt(p) for p in papers]
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores: list[float] = []
        for out in outputs:
            # 1-token generation; .logprobs is a list of dicts {token_id: Logprob}
            top_logprobs = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            la = top_logprobs.get(self.accept_id)
            lr = top_logprobs.get(self.reject_id)
            # vLLM may surface the float on .logprob or as the dict value directly
            la_v = float(la.logprob) if hasattr(la, "logprob") else (float(la) if la is not None else -50.0)
            lr_v = float(lr.logprob) if hasattr(lr, "logprob") else (float(lr) if lr is not None else -50.0)
            scores.append(la_v - lr_v)
        return scores

    def close(self) -> None:
        # vLLM cleans up on GC; provided for symmetry
        self.llm = None
