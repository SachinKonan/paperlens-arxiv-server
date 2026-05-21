"""End-to-end test against a running paperlens-arxiv-server.

Skipped unless ``PAPERLENS_E2E_URL`` is set (e.g. ``http://localhost:8000``).
Requires the upstream arxiv_retriever to also be reachable per the running
server's config.
"""
from __future__ import annotations

import os
import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.environ.get("PAPERLENS_E2E_URL"),
    reason="set PAPERLENS_E2E_URL (e.g. http://localhost:8000) to run",
)


def test_health():
    url = os.environ["PAPERLENS_E2E_URL"].rstrip("/")
    r = requests.get(f"{url}/health", timeout=10)
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    assert "reranker" in body
    assert "retriever" in body


def test_search_rerank_changes_order():
    url = os.environ["PAPERLENS_E2E_URL"].rstrip("/")
    payload = {
        "query": "sparse attention transformer",
        "topk_retrieve": 20,
        "topk_rerank": 10,
        "upper_bound_datetime": "2024-01-01",
    }
    r = requests.post(f"{url}/search", json=payload, timeout=120)
    r.raise_for_status()
    body = r.json()

    assert body["n_returned"] == min(10, body["n_retrieved"])
    assert len(body["results"]) == body["n_returned"]

    # The reranker should be doing _something_: at least one paper should not
    # be at the same (rerank_position, retriever_position).
    moved = [p for p in body["results"]
             if p["rerank_position"] != p["retriever_position"]]
    assert moved, "reranker left ordering identical -- check reranker config"

    # accept_score should be monotonically non-increasing in rerank_position
    scores = [p["accept_score"] for p in body["results"]]
    assert scores == sorted(scores, reverse=True), "results not sorted by accept_score"
