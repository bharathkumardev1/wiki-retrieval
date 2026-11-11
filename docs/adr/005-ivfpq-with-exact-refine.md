# 005. IVF-PQ candidates, re-scored with full-precision vectors

Status: accepted for the dev corpus; to be re-measured at full-corpus scale

## Context
On the full NQ corpus, IVF-PQ alone loses a lot: PQ96 at nprobe 64 scores nDCG@10 0.372 against 0.402 for exact search (results/gpu_full_nq_faiss_lora_easy_t4.json). On the 104,201 passage dev slice with PQ48 and nlist 1024, the raw index at nprobe 16 loses 0.128 nDCG@10 against exact dense search.

## What I tried (dev slice, 3,452 queries, LoRA-tuned encoder; results/dev_index_lora_easy*.json)
The compressed index returns `n_candidates` passages and the dense ordering is recomputed with exact inner products on the float32 vectors.

| nprobe | candidates | overlap@100 with exact | hybrid nDCG@10 (exact: 0.7856) | dense stage p50 |
|---|---|---|---|---|
| 16 | 200 | 0.669 | 0.7244 | 0.31 ms |
| 64 | 500 | 0.879 | 0.7676 | 0.56 ms |
| 128 | 1000 | 0.946 | 0.7773 | 0.95 ms |
| 256 | 1000 | 0.978 | 0.7831 | 1.26 ms |

Brute-force exact search on the same host takes 12.3 ms p50 (numpy matmul, one query).

## Decision
Default to nprobe 256 and 1000 candidates. At that point the paired nDCG@10 loss against exact hybrid search is 0.0025 [0.0007, 0.0044]: small, but the interval excludes zero, so I report it as a loss. Without the refine step the same index loses 0.0185 [0.0139, 0.0232] (hybrid), so the refine step recovers most of the damage. The operating point was chosen by overlap with exact search, not by relevance labels, to avoid tuning on the evaluation judgments.

## Limits
- The 10x dense-stage speedup is measured at 104k passages. At 2.68M passages the exact scan is about 26 times larger, so the gap should widen, but I have not measured it there.
- Refine needs the float vectors in memory (4.1 GB for the full corpus in float32, 2.1 GB in float16), so the compression speeds up search and does not shrink the process.
- nprobe 256 of 1024 lists is a quarter of the index; at 4096 lists on the full corpus the right setting will differ.
- Run on a 2 vCPU container, one query at a time.
