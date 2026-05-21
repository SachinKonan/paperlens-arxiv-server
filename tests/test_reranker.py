"""Smoke test: load the vision reranker against a real PaperLens ckpt and
score two papers. Confirms vLLM loading + DECISION_TOKEN_IDX=5 extraction +
p_accept softmax produce sensible values.

Heavy: loads ~6 GB of weights into vLLM. Skipped unless env vars are set.

Env:
  PAPERLENS_TEST_CKPT_PATH   local path or HF repo id of a PaperLens ckpt
  PAPERLENS_TEST_MODALITY    'text' or 'vision' (default 'text' -- no images needed)
  PAPERLENS_TEST_IMAGES_DIR  required only when modality=vision; dir with page_*.png
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

CKPT = os.environ.get("PAPERLENS_TEST_CKPT_PATH")
MODALITY = os.environ.get("PAPERLENS_TEST_MODALITY", "text")
IMAGES_DIR = os.environ.get("PAPERLENS_TEST_IMAGES_DIR")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="set PAPERLENS_TEST_CKPT_PATH to a PaperLens ckpt to run",
)


def test_score_two_papers():
    from paperlens_arxiv_server.reranker import PaperLensReranker
    from paperlens_arxiv_server.retriever_client import RetrievedPaper

    if MODALITY == "vision":
        assert IMAGES_DIR, "vision mode requires PAPERLENS_TEST_IMAGES_DIR"
        from PIL import Image
        pngs = sorted(Path(IMAGES_DIR).glob("page_*.png"))
        assert pngs, f"no page_*.png in {IMAGES_DIR}"
        images = [Image.open(p) for p in pngs[:8]]
    else:
        images = []

    rr = PaperLensReranker(
        ckpt_path=CKPT, modality=MODALITY, domain="arxiv",
        gpu_memory_utilization=0.7, max_model_len=8192,
    )

    papers = [
        RetrievedPaper(
            paper_id="fake-strong",
            title="Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
            abstract=(
                "Transformers have a potential of learning longer-term dependency, but "
                "are limited by a fixed-length context. We propose Transformer-XL that "
                "enables learning dependency beyond a fixed length without disrupting "
                "temporal coherence."
            ),
            score=0.95,
            images=list(images),
        ),
        RetrievedPaper(
            paper_id="fake-weak",
            title="A Small Modification of Softmax",
            abstract=(
                "In this paper we propose a small modification of the softmax function. "
                "We test it on MNIST and observe a 0.1% accuracy improvement, though "
                "the result is within noise."
            ),
            score=0.94,
            images=list(images),
        ),
    ]
    scores = rr.score(papers)
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0, f"p_accept {s} not in [0,1]"
    # Sanity: a trained reranker should rate Transformer-XL higher than a weak foil
    assert scores[0] > scores[1] - 0.05, (
        f"expected STRONG > WEAK (or close); got {scores[0]:.3f} vs {scores[1]:.3f}"
    )
