"""FastAPI server: POST /search -> retrieve 200 -> rerank with cached p_accept -> top-k.

Flow per RANKER.md and user-confirmed deployment spec:

    user -> POST /search {query, k<=200}
      retriever.retrieve(query, topk=200)            # always 200 from upstream
      cache.get(ckpt_id, [arxiv_id for _ in 200])    # partition into hit / miss
      reranker.score(papers_miss)                    # vLLM vision inference on misses only
      cache.put(ckpt_id, miss_scores)
      blend = 0.3 * z(retriever_score) + 0.7 * p_accept     (RANKER.md §5.2)
      sort by blend, return top-k
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from .cache import PAcceptCache, hash_ckpt_id
from .image_loader import resolve_images_root, resolve_page_paths
from .reranker import PaperLensReranker
from .retriever_client import ArxivRetrieverClient, RetrievedPaper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("paperlens-arxiv-server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=200, description="Number of reranked results to return (<=200)")
    upper_bound_datetime: Optional[str] = Field(
        default=None,
        description="ISO date; arxiv_retriever filters to papers submitted on/before this date",
    )
    exclude_title: Optional[str] = Field(
        default=None,
        description="Fuzzy-match title to exclude from results",
    )
    blend: Optional[str] = Field(
        default=None,
        description="'default' = 0.3*z(retriever)+0.7*p_accept (RANKER.md §5.2). "
                    "'p_accept_only' = rank by p_accept alone.",
    )


class RankedPaper(BaseModel):
    paper_id: str                       # = arxiv_id
    title: str
    abstract: str
    p_accept: float                     # softmax([logp_accept, logp_reject]) in [0,1]
    retriever_score: float              # raw upstream score (cosine for 0.6B-pv; BM25 score for bm25-pv)
    blend_score: float                  # final ranking score
    rerank_position: int                # 0-indexed in returned results
    retriever_position: int             # 0-indexed in upstream top-200
    cache_hit: bool                     # True if p_accept came from cache, False if computed fresh
    submission_date: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    n_retrieved: int
    n_returned: int
    n_cache_hits: int                   # of the 200, how many had a cached p_accept
    n_inferred: int                     # of the 200, how many we ran through vLLM this request
    results: list[RankedPaper]


# ---------------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------------

app = FastAPI(title="paperlens-arxiv-server")
_state: dict = {}


def _zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in xs]


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("PAPERLENS_SERVER_CONFIG", "configs/server.yaml")
    cfg = OmegaConf.load(cfg_path)
    log.info(f"loaded config from {cfg_path}")
    log.info(OmegaConf.to_yaml(cfg))

    _state["cfg"] = cfg
    _state["images_root"] = resolve_images_root(cfg.get("images_root"))
    # ckpt_id is now derived from paperlens_serve's reported ckpt_path,
    # not a local file (we don't load the model in this process).
    _state["retriever"] = ArxivRetrieverClient(
        base_url=cfg.retriever.base_url,
        timeout=cfg.retriever.timeout_seconds,
    )
    _state["reranker"] = PaperLensReranker(
        serve_url=cfg.paperlens_serve.base_url,
        domain=cfg.reranker.domain,
        modality=cfg.reranker.modality,
        timeout_seconds=float(cfg.paperlens_serve.get("timeout_seconds", 600)),
    )
    # Pull the upstream serve's ckpt_path so the cache key matches what
    # paperlens-serve actually loaded (decouples our config from theirs).
    try:
        h = requests.get(f"{cfg.paperlens_serve.base_url}/health", timeout=10).json()
        _state["ckpt_id"] = hash_ckpt_id(h.get("ckpt_path", cfg.paperlens_serve.base_url))
        _state["compute_arch"] = h.get("compute_arch", "unknown")
    except Exception as e:
        log.warning(f"serve /health probe failed at startup: {e}")
        _state["ckpt_id"] = hash_ckpt_id(cfg.paperlens_serve.base_url)
        _state["compute_arch"] = "unknown"

    _state["cache"] = PAcceptCache(cfg.cache.db_path)
    log.info(
        f"startup complete. ckpt_id={_state['ckpt_id']} "
        f"images_root={_state['images_root']} "
        f"compute_arch={_state['compute_arch']} "
        f"cache_rows={_state['cache'].size(_state['ckpt_id'])}"
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    try:
        _state["cache"].close()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    cfg = _state["cfg"]
    return {
        "status": "ok",
        "paperlens_serve_url": cfg.paperlens_serve.base_url,
        "ckpt_id": _state["ckpt_id"],
        "compute_arch": _state["compute_arch"],
        "retriever_url": cfg.retriever.base_url,
        "images_root": str(_state["images_root"]),
        "cache_db": cfg.cache.db_path,
        "cache_rows_for_ckpt": _state["cache"].size(_state["ckpt_id"]),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    cfg = _state["cfg"]
    topk_retrieve = int(cfg.retriever.topk_retrieve)

    log.info(f"query={req.query!r} k={req.k} blend={req.blend or 'default'}")

    # 1) Retrieve fixed top-200
    try:
        papers = _state["retriever"].retrieve(
            query=req.query,
            topk=topk_retrieve,
            upper_bound_datetime=req.upper_bound_datetime,
            exclude_title=req.exclude_title,
        )
    except Exception as e:
        log.exception("retriever failed")
        raise HTTPException(502, f"arxiv_retriever error: {e}")
    if not papers:
        return SearchResponse(
            query=req.query, n_retrieved=0, n_returned=0,
            n_cache_hits=0, n_inferred=0, results=[],
        )

    # 2) Partition into cache hits + misses
    arxiv_ids = [p.paper_id for p in papers]
    cached = _state["cache"].get(_state["ckpt_id"], arxiv_ids)
    miss_papers = [p for p in papers if p.paper_id not in cached]
    log.info(
        f"retrieved {len(papers)}, cache hits {len(cached)}, "
        f"misses {len(miss_papers)} (will run vLLM)"
    )

    # 3) For misses: attach image PATHS (the upstream `paperlens serve` does
    #    the actual image load + LF tokenization). Then score via /score.
    if miss_papers:
        for p in miss_papers:
            p.images = resolve_page_paths(p.paper_id, _state["images_root"])
        miss_scores_list = _state["reranker"].score(miss_papers)
        miss_scores = {
            p.paper_id: s for p, s in zip(miss_papers, miss_scores_list)
        }
        _state["cache"].put(
            _state["ckpt_id"], miss_scores,
            source="online", compute_arch=_state["compute_arch"],
        )
    else:
        miss_scores = {}

    p_accept_by_id: dict[str, float] = {**cached, **miss_scores}

    # 4) Compute blended ranking score
    retriever_scores = [p.score for p in papers]
    z = _zscore(retriever_scores)
    blend_mode = (req.blend or cfg.search.get("default_blend", "default")).lower()
    blends: list[float] = []
    if blend_mode == "p_accept_only":
        blends = [p_accept_by_id.get(p.paper_id, 0.5) for p in papers]
    else:  # 'default' or anything else -> documented blend
        wr = float(cfg.search.blend_weight_retriever)
        wp = float(cfg.search.blend_weight_p_accept)
        for zi, p in zip(z, papers):
            pa = p_accept_by_id.get(p.paper_id, 0.5)
            blends.append(wr * zi + wp * pa)

    # 5) Sort by blend desc, return top-k
    ranked = sorted(
        enumerate(zip(papers, blends)),
        key=lambda kv: -kv[1][1],
    )
    results = []
    for new_pos, (retr_pos, (p, b)) in enumerate(ranked[: req.k]):
        results.append(RankedPaper(
            paper_id=p.paper_id,
            title=p.title,
            abstract=p.abstract,
            p_accept=p_accept_by_id.get(p.paper_id, 0.5),
            retriever_score=p.score,
            blend_score=b,
            rerank_position=new_pos,
            retriever_position=retr_pos,
            cache_hit=p.paper_id in cached,
            submission_date=p.submission_date,
        ))
    return SearchResponse(
        query=req.query,
        n_retrieved=len(papers),
        n_returned=len(results),
        n_cache_hits=len(cached),
        n_inferred=len(miss_papers),
        results=results,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
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
