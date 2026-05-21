"""FastAPI server: POST /search -> retrieve K -> rerank with PaperLens -> top N."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from .prompts import prompt_for
from .reranker import PaperLensReranker
from .retriever_client import ArxivRetrieverClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
log = logging.getLogger("paperlens-arxiv-server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    topk_retrieve: Optional[int] = Field(default=None,
        description="How many candidates to fetch from arxiv_retriever (defaults to config search.default_topk_retrieve)")
    topk_rerank: Optional[int] = Field(default=None,
        description="How many ranked results to return (defaults to config search.default_topk_rerank)")
    upper_bound_datetime: Optional[str] = Field(default=None,
        description="ISO date; arxiv_retriever filters to papers submitted on/before this date")
    exclude_title: Optional[str] = Field(default=None,
        description="Fuzzy-match title to exclude from results")


class RankedPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    accept_score: float                   # logprob(Accept) - logprob(Reject), higher = better
    retriever_score: float                # cosine sim from upstream FAISS
    rerank_position: int                  # 0-indexed position after reranking
    retriever_position: int               # 0-indexed position before reranking
    submission_date: Optional[str] = None
    arxiv_id: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    n_retrieved: int
    n_returned: int
    results: list[RankedPaper]


# ---------------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------------

app = FastAPI(title="paperlens-arxiv-server")
_state: dict = {}


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("PAPERLENS_SERVER_CONFIG", "configs/server.yaml")
    cfg = OmegaConf.load(cfg_path)
    log.info(f"loaded config from {cfg_path}")
    log.info(OmegaConf.to_yaml(cfg))

    _state["cfg"] = cfg
    _state["retriever"] = ArxivRetrieverClient(
        base_url=cfg.retriever.base_url,
        timeout=cfg.retriever.timeout_seconds,
    )
    _state["reranker"] = PaperLensReranker(
        hf_repo=cfg.reranker.hf_repo,
        modality=cfg.reranker.modality,
        domain=cfg.reranker.domain,
        tensor_parallel_size=cfg.reranker.tensor_parallel_size,
        gpu_memory_utilization=cfg.reranker.gpu_memory_utilization,
        max_model_len=cfg.reranker.max_model_len,
    )
    log.info("startup complete")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "reranker": _state["cfg"].reranker.hf_repo,
        "retriever": _state["cfg"].retriever.base_url,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    cfg = _state["cfg"]
    topk_r = req.topk_retrieve or cfg.search.default_topk_retrieve
    topk_n = req.topk_rerank or cfg.search.default_topk_rerank
    if topk_r > cfg.search.max_topk_retrieve:
        raise HTTPException(400, f"topk_retrieve {topk_r} > max {cfg.search.max_topk_retrieve}")
    if topk_n > topk_r:
        raise HTTPException(400, "topk_rerank cannot exceed topk_retrieve")

    log.info(f"query={req.query!r} retrieve={topk_r} rerank={topk_n}")

    try:
        papers = _state["retriever"].retrieve(
            query=req.query, topk=topk_r,
            upper_bound_datetime=req.upper_bound_datetime,
            exclude_title=req.exclude_title,
        )
    except Exception as e:
        log.exception("retriever failed")
        raise HTTPException(502, f"arxiv_retriever error: {e}")

    if not papers:
        return SearchResponse(query=req.query, n_retrieved=0, n_returned=0, results=[])

    try:
        scores = _state["reranker"].score(papers)
    except Exception as e:
        log.exception("reranker failed")
        raise HTTPException(500, f"reranker error: {e}")

    enumerated = list(enumerate(zip(papers, scores)))
    enumerated.sort(key=lambda kv: kv[1][1], reverse=True)

    results = []
    for new_pos, (retr_pos, (p, s)) in enumerate(enumerated[:topk_n]):
        results.append(RankedPaper(
            paper_id=p.paper_id, title=p.title, abstract=p.abstract,
            accept_score=s, retriever_score=p.score,
            rerank_position=new_pos, retriever_position=retr_pos,
            submission_date=p.submission_date, arxiv_id=p.arxiv_id,
        ))
    return SearchResponse(
        query=req.query, n_retrieved=len(papers), n_returned=len(results), results=results,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/server.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    os.environ["PAPERLENS_SERVER_CONFIG"] = args.config
    cfg = OmegaConf.load(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
