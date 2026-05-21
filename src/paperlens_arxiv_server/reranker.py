"""PaperLens reranker: score(papers) -> list[float].

This mirrors the lab's inference pipeline exactly (see scripts/vllm_infer.py
in paperlens-training-and-inference, modified to record logprob arrays):

  1. Build the canonical ShareGPT messages [system, user_with_paper].
  2. Apply the model's chat template with ``add_generation_prompt=True``
     so the assistant turn starts fresh.
  3. Sample (greedy) up to ``max_new_tokens=8`` tokens with ``logprobs=5``
     so we capture the top-5 token logprobs at every step.
  4. The decision token sits at **position ``DECISION_TOKEN_IDX=5``** of the
     generated sequence: ``Outcome``, ``:``, `` \\``, ``boxed``, ``{``,
     ``Accept|Reject``.
  5. Score = ``logprob_accept[5] - logprob_reject[5]`` — the same formula
     that drives ``iclr_calibrated_acc_2526.py`` and ``arxiv_calibrated_acc.py``.

Higher score => more likely to be accepted at a top venue.
"""
from __future__ import annotations

import logging
from typing import Optional

from .prompts import SHAREGPT_SYSTEM_PROMPT, build_user_turn
from .retriever_client import RetrievedPaper


log = logging.getLogger(__name__)

# Matches scripts/iclr_calibrated_acc_2526.py and main repo vllm_infer.py:
# "Outcome:" + " " + "\\" + "boxed" + "{" -- 5 tokens before X under
# the Qwen2.5 BPE tokenizer.
DECISION_TOKEN_IDX = 5
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
        # Greedy decode through the boxed-decision token.
        # ``logprobs=5`` matches scripts/vllm_infer.py; "Accept" and "Reject"
        # are reliably in the top-5 at position 5 for any trained PaperLens.
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            logprobs=5,
        )

        # Verify Accept / Reject each tokenize to a single token. The lab's
        # vllm_infer.py asserts this same invariant -- if it ever fires we
        # need to update DECISION_TOKEN_IDX and the score-extraction logic.
        accept_ids = self.tokenizer.encode(ACCEPT_TOKEN, add_special_tokens=False)
        reject_ids = self.tokenizer.encode(REJECT_TOKEN, add_special_tokens=False)
        assert len(accept_ids) == 1, f"'Accept' encodes to {len(accept_ids)} tokens: {accept_ids!r} (expected 1)"
        assert len(reject_ids) == 1, f"'Reject' encodes to {len(reject_ids)} tokens: {reject_ids!r} (expected 1)"
        self.accept_id = accept_ids[0]
        self.reject_id = reject_ids[0]
        log.info(f"decision token IDs: Accept={self.accept_id}, Reject={self.reject_id}")

    def _build_messages(self, paper: RetrievedPaper) -> list[dict]:
        """Compose the ShareGPT-style messages list for one paper.

        Returns ``[system, user]`` -- the model generates the assistant turn
        autoregressively so we can record per-step logprobs (matches the lab
        pipeline; do NOT prefill the assistant turn).
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
        ]

    def _render_prompt(self, paper: RetrievedPaper) -> str:
        """Apply the chat template with the model's normal generation prefix."""
        messages = self._build_messages(paper)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def score(self, papers: list[RetrievedPaper]) -> list[float]:
        """Return ``logprob_accept[5] - logprob_reject[5]`` per paper.

        Position 5 is the literal Accept|Reject token in
        ``Outcome: \\boxed{Accept|Reject}``. If the generation is truncated
        before position 5 (rare for a trained model) the score falls back
        to ``0.0`` (= no preference).
        """
        if not papers:
            return []
        prompts = [self._render_prompt(p) for p in papers]
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores: list[float] = []
        for out in outputs:
            per_step_logprobs = out.outputs[0].logprobs or []
            if len(per_step_logprobs) <= DECISION_TOKEN_IDX:
                log.warning(f"generation too short ({len(per_step_logprobs)} steps); scoring 0.0")
                scores.append(0.0)
                continue
            step_lp = per_step_logprobs[DECISION_TOKEN_IDX]
            la = step_lp.get(self.accept_id) if step_lp else None
            lr = step_lp.get(self.reject_id) if step_lp else None
            # vLLM surfaces a Logprob object with .logprob; tolerate raw floats too.
            la_v = float(la.logprob) if hasattr(la, "logprob") else (float(la) if la is not None else -50.0)
            lr_v = float(lr.logprob) if hasattr(lr, "logprob") else (float(lr) if lr is not None else -50.0)
            scores.append(la_v - lr_v)
        return scores

    def close(self) -> None:
        # vLLM cleans up on GC; provided for symmetry
        self.llm = None
