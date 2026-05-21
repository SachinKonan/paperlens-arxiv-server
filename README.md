# paperlens-arxiv-server

PaperLens **vision** reranker stacked on top of [arxiv_retriever](https://github.com/SachinKonan/arxiv_retriever) —
serves arXiv search results sorted by `0.3·z(retriever_score) + 0.7·p_accept`
where `p_accept` is PaperLens-V's softmax([logp_accept, logp_reject]) over the
boxed-decision token. Mirrors the litsearch reranking experiment in
[`paperlens-training-and-inference/tmp_latex_dir/RANKER.md`](../paperlens-training-and-inference/docs).

```
user → POST /search {query, k <= 200}
  paperlens-arxiv-server :8000
    1) retriever.retrieve(query, topk=200)            → 200 candidates by FAISS cosine
    2) cache.get(ckpt_id, arxiv_ids)                  → partition into hits / misses
    3) reranker.score(misses) on H100/A100 (vLLM)     → p_accept for new papers
    4) cache.put(misses) and read-through next time
    5) rank = 0.3·z(retriever_score) + 0.7·p_accept   (per RANKER.md §5.2)
    → return top-k
```

The default reranker is **PaperLens-V-3B** at the local ckpt
`saves/.../arxiv_train/small/arxiv_21k_vision_3b/checkpoint-5236` (or the
published HF repo `skonan/paperlens-3b-vision-arxiv`). Override via
`configs/server.yaml`.

The default retriever is **Qwen3-Embedding-0.6B FAISS over the per_venue 80K
subset** — same indexes used in the offline litsearch reranking eval.

---

## Layout

```
paperlens-arxiv-server/
├── pyproject.toml
├── configs/server.yaml             # reranker ckpt, retriever URL, blend weights, cache db
├── scripts/
│   ├── launch_local.sh             # spins up arxiv_retriever + this server
│   └── bootstrap_cache_from_litsearch.py   # pre-load p_accept from RANKER.md §6.1 parquet
├── src/paperlens_arxiv_server/
│   ├── server.py                   # FastAPI /search + /health
│   ├── reranker.py                 # vLLM-loaded PaperLens-V; p_accept softmax
│   ├── retriever_client.py         # HTTP client for arxiv_retriever
│   ├── image_loader.py             # data/images_arxiv/<arxiv_id>/page_*.png → PIL list
│   ├── cache.py                    # SQLite p_accept cache (online + bootstrap)
│   └── prompts.py                  # PROMPT_ARXIV / PROMPT_ICLR
├── tests/{test_reranker.py, test_e2e.py, test_cache.py}
└── external/arxiv_retriever/       # git submodule (per_venue configs added to configs/retrieval/)
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
# 1) (one-time) Pre-load the cache from the offline litsearch predictions
#    -> 28,664 papers cached for ~36% of the per_venue 80K corpus before the first query
python scripts/bootstrap_cache_from_litsearch.py \
    --parquet /scratch/gpfs/ZHUANGL/sk7524/litsearch_eval/passover/predictions_3b.parquet \
    --ckpt_path /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/saves/.../arxiv_21k_vision_3b/checkpoint-5236 \
    --db_path ./cache/p_accept.sqlite

# 2) Start both services
bash scripts/launch_local.sh                  # foreground, or PAPERLENS_BG=1 for bg

# Or run them manually:
#   (in conda) bash external/arxiv_retriever/src/arxiv_retriever/server/retrieval_launch.sh \
#                      external/arxiv_retriever/configs/retrieval/qwen3_06b_per_venue.yaml \
#                      server.port=8001
#   (in uv)    paperlens-arxiv-server --config configs/server.yaml
```

Smoke check:

```bash
curl -sf http://localhost:8000/health | jq

curl -sf -X POST http://localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "sparse attention transformer",
  "k": 10,
  "upper_bound_datetime": "2024-01-01"
}' | jq
```

Response format:

```json
{
  "query": "sparse attention transformer",
  "n_retrieved": 200,
  "n_returned": 10,
  "n_cache_hits": 174,
  "n_inferred": 26,
  "results": [
    {
      "paper_id": "2311.04567",
      "title": "...",
      "abstract": "...",
      "p_accept": 0.82,            // softmax([logp_accept, logp_reject]) in [0,1]
      "retriever_score": 0.71,
      "blend_score": 0.71,         // 0.3 * z(retriever_score) + 0.7 * p_accept
      "rerank_position": 0,
      "retriever_position": 11,
      "cache_hit": true,
      "submission_date": "2023-11-12"
    }
  ]
}
```

`n_inferred` is how many of the 200 retrieved papers needed fresh vLLM inference;
the rest came from the SQLite cache. A repeated identical query will return
`n_inferred=0` and respond sub-second.

---

## Tests

```bash
# Cache unit tests (pure Python, no GPU)
uv run pytest tests/test_cache.py -v

# Reranker smoke (loads vLLM, ~1 min on H100; vision mode needs sample images)
PAPERLENS_TEST_CKPT_PATH=<local path or HF repo> \
PAPERLENS_TEST_MODALITY=text \
  uv run pytest tests/test_reranker.py -v

# End-to-end (assumes both services are running per scripts/launch_local.sh)
PAPERLENS_E2E_URL=http://localhost:8000 \
  uv run pytest tests/test_e2e.py -v
```

## Static assets

The server reads (never writes) these external resources -- collect them once
and point the config at them:

| Asset | Default path | Owner |
|---|---|---|
| BM25 per_venue Lucene index | `/scratch/gpfs/ZHUANGL/sk7524/arxiv_bm25_per_venue_index/` | arxiv_retriever |
| FAISS per_venue qwen3-0.6b | `/scratch/gpfs/ZHUANGL/sk7524/arxiv_faiss_per_venue_index/qwen3_Flat.index` | arxiv_retriever |
| arxiv metadata snapshot | `/scratch/gpfs/ZHUANGL/sk7524/SkyRL/.../arxiv-metadata-oai-snapshot.jsonl` | upstream OAI |
| Per-paper page PNGs | `${PAPERLENS_IMAGES_ROOT}/<arxiv_id>/page_N.png` | `reconstruction.py` (or LF main repo) |
| Reranker ckpt | from `configs/server.yaml::reranker.ckpt_path` | published HF model |
| Cache bootstrap parquet | `/scratch/gpfs/ZHUANGL/sk7524/litsearch_eval/passover/predictions_3b.parquet` | RANKER.md §6.1 |

The server creates and mutates only:

| Asset | Default path | Notes |
|---|---|---|
| p_accept cache | `./cache/p_accept.sqlite` | Read-through; bootstrap once from the parquet above |

### Image-path coherence across the ecosystem

This server, `paperlens-training-and-inference` (which provides `reconstruction.py`),
and the published HF dataset all key on the same `arxiv_id`. Two env vars line them up:

- `PAPERLENS_DATA_ROOT` → wherever reconstruction.py wrote the rebuilt data tree
- `PAPERLENS_IMAGES_ROOT` → defaults to `${PAPERLENS_DATA_ROOT}/images_arxiv`

## Known caveat: cross-architecture p_accept drift

vLLM picks attention kernels per-GPU-class (the torch.compile cache hash differs
across H100-SXM / gpu80 / A100-PCIE). For the **same model + same inputs + same
vLLM/transformers version**, bf16 logits drift across GPU classes by:

| Statistic | Value (1000-row diag: PLI H100-SXM vs gputest gpu80) |
|---|---|
| Mean   \|Δp_accept\| | 0.022 |
| Median \|Δp_accept\| | 0.010 |
| Max    \|Δp_accept\| | 0.214 (concentrated at p≈0.5) |
| Binary Accept/Reject flips | 1.9 % |
| Pearson r(p_archA, p_archB) | **0.994** |

Implications for the deployment:

- **Top-K reranking is preserved**: ranks are virtually identical (Pearson > 0.99) so
  RANKER.md's recall@K numbers stay valid across architectures.
- **Absolute p_accept values can differ** by up to 0.21 on borderline papers (those
  sitting around p≈0.5 where small logit shifts flip the softmax).
- **Cache hygiene**: every row in the `p_accept.sqlite` cache is tagged with a
  `compute_arch` column. Rows bootstrapped from `predictions_3b.parquet` are
  marked `pli_h100_sxm` (the May-12 hardware); rows produced online are tagged
  with the running server's hardware. Mixed-arch caches are operationally fine
  but auditable — see `SELECT compute_arch, COUNT(*) FROM p_accept GROUP BY 1`.

We chose to **bootstrap from PLI and serve on gpu-test** intentionally: the
PLI parquet covers 28k papers for free (no GPU cost) and the gpu-test online
backfill is fast. The 0.994 correlation means the cache and the live scorer
agree on ranking even though they differ in 3rd-decimal absolute values.
