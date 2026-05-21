"""Thin HTTP client around arxiv_retriever's POST /retrieve endpoint."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests


log = logging.getLogger(__name__)


@dataclass
class RetrievedPaper:
    paper_id: str
    title: str
    abstract: str
    score: float          # cosine similarity from the upstream FAISS index
    full_text: Optional[str] = None
    submission_date: Optional[str] = None
    arxiv_id: Optional[str] = None


class ArxivRetrieverClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, query: str, topk: int = 50,
                 upper_bound_datetime: Optional[str] = None,
                 exclude_title: Optional[str] = None) -> list[RetrievedPaper]:
        payload = {"query": query, "topk": topk}
        if upper_bound_datetime is not None:
            payload["upper_bound_datetime"] = upper_bound_datetime
        if exclude_title is not None:
            payload["exclude_title"] = exclude_title

        url = f"{self.base_url}/retrieve"
        log.info(f"POST {url} topk={topk} upper={upper_bound_datetime}")
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        results = body.get("results", body) if isinstance(body, dict) else body
        return [self._row_to_paper(row) for row in results]

    @staticmethod
    def _row_to_paper(row: dict) -> RetrievedPaper:
        # The upstream server returns canonical fields; tolerate field-name drift.
        return RetrievedPaper(
            paper_id=str(row.get("paper_id") or row.get("arxiv_id") or row.get("id", "")),
            title=str(row.get("title", "")).strip(),
            abstract=str(row.get("abstract", "")).strip(),
            score=float(row.get("score", 0.0)),
            full_text=row.get("full_text") or row.get("body") or row.get("content"),
            submission_date=row.get("submission_date") or row.get("date"),
            arxiv_id=row.get("arxiv_id"),
        )
