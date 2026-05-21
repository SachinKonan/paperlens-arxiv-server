# paperlens-arxiv-server

PaperLens reranker stacked on top of [arxiv_retriever](https://github.com/SachinKonan/arxiv_retriever) —
returns arXiv search results sorted by **predicted acceptance probability** (the
PaperLens accept-vs-reject logit gap) instead of raw cosine similarity.

```
client → POST /search {query, topk_retrieve=50, topk_rerank=10, upper_bound_datetime?}
  paperlens-arxiv-server :8000
    ├── arxiv_retriever HTTP :8001  → top-K by FAISS cosine
    └── PaperLens vLLM (3B-text-arxiv default)
            score = logprob(Accept) − logprob(Reject)
  → sort by score desc, return top-N
```

The default reranker model is the smallest PaperLens repo
([`skonan/paperlens-3b-text-arxiv`](https://huggingface.co/skonan/paperlens-3b-text-arxiv));
override via `configs/server.yaml` to use any of the other 7
([PaperLens HF collection](https://huggingface.co/collections/skonan/paperlens-6a0c79da423c3a436b7f6b1a)).

---

## Layout

```
paperlens-arxiv-server/
├── pyproject.toml
├── configs/server.yaml          # reranker model, ports, top-K defaults
├── scripts/launch_local.sh      # spins up arxiv_retriever + this server
├── src/paperlens_arxiv_server/
│   ├── server.py                # FastAPI /search + /health
│   ├── reranker.py              # vLLM-loaded PaperLens scorer
│   ├── retriever_client.py      # HTTP client for arxiv_retriever
│   └── prompts.py               # Verbatim PROMPT_ARXIV / PROMPT_ICLR
├── tests/{test_reranker.py, test_e2e.py}
└── external/arxiv_retriever/    # git submodule
```

---

## Install

The upstream `arxiv_retriever` requires `faiss-gpu` (conda-only), so it has
its own conda env. This server is independent and runs in a `uv`/`pip` env.

```bash
# 1) clone with submodule
git clone --recurse-submodules <this repo> paperlens-arxiv-server
cd paperlens-arxiv-server

# 2) arxiv_retriever: conda env per its README
cd external/arxiv_retriever
conda env create -f conda-env.yml      # creates env `retriever`
cd ../..

# 3) paperlens-arxiv-server: uv env
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Hardware: the reranker loads via vLLM; one A100/H100 GPU is enough for the
3B default. Override `reranker.gpu_memory_utilization` in `configs/server.yaml`
to share a GPU with the retriever.

---

## Run

```bash
# Both services in one go (foreground); set PAPERLENS_BG=1 for background.
bash scripts/launch_local.sh

# Or run manually:
#   (in conda) bash external/arxiv_retriever/src/arxiv_retriever/server/retrieval_launch.sh ...
#   (in uv)    paperlens-arxiv-server --config configs/server.yaml
```

Smoke check:

```bash
curl -sf http://localhost:8000/health | jq

curl -sf -X POST http://localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "sparse attention transformer",
  "topk_retrieve": 20,
  "topk_rerank": 5,
  "upper_bound_datetime": "2024-01-01"
}' | jq
```

Response format:

```json
{
  "query": "...",
  "n_retrieved": 20,
  "n_returned": 5,
  "results": [
    {
      "paper_id": "...",
      "title": "...",
      "abstract": "...",
      "accept_score": 2.34,         // logprob(Accept) − logprob(Reject)
      "retriever_score": 0.87,
      "rerank_position": 0,
      "retriever_position": 11,
      "submission_date": "2023-11-12",
      "arxiv_id": "2311.04567"
    }
  ]
}
```

---

## Tests

```bash
# Reranker smoke (loads vLLM, ~30s)
PAPERLENS_TEST_HF_REPO=skonan/paperlens-3b-text-arxiv \
  uv run pytest tests/test_reranker.py -v

# End-to-end (assumes both services are running per scripts/launch_local.sh)
PAPERLENS_E2E_URL=http://localhost:8000 \
  uv run pytest tests/test_e2e.py -v
```
