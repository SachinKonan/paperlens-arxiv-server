"""Smoke test: load the reranker, score two fabricated papers, expect a
finite logprob-difference score for each.

Skipped unless ``PAPERLENS_TEST_HF_REPO`` is set (loading vLLM is heavy).
"""
from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PAPERLENS_TEST_HF_REPO"),
    reason="set PAPERLENS_TEST_HF_REPO to a PaperLens HF repo id to run",
)


def test_score_two_papers():
    from paperlens_arxiv_server.reranker import PaperLensReranker
    from paperlens_arxiv_server.retriever_client import RetrievedPaper

    hf_repo = os.environ["PAPERLENS_TEST_HF_REPO"]
    rr = PaperLensReranker(hf_repo=hf_repo, modality="text", domain="arxiv")

    papers = [
        RetrievedPaper(
            paper_id="fake-1",
            title="A Strong Result on Attention",
            abstract="We present a novel sparse attention mechanism that achieves "
                     "state-of-the-art results on multiple benchmarks. Our method ...",
            score=0.95,
            full_text="# A Strong Result on Attention\n\nAnonymous Submission\n\n"
                      "# Abstract\nWe present a novel sparse attention mechanism ...",
        ),
        RetrievedPaper(
            paper_id="fake-2",
            title="A Weak Result on Attention",
            abstract="In this paper we propose a marginal extension of softmax. Results "
                     "are inconclusive and the experimental section is sparse ...",
            score=0.92,
            full_text="# A Weak Result on Attention\n\nAnonymous Submission\n\n"
                      "# Abstract\nIn this paper we propose a marginal extension ...",
        ),
    ]
    scores = rr.score(papers)
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, float)
        assert -50.0 < s < 50.0, f"score {s} suspiciously out of range"
