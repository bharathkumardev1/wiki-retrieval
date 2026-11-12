# 007. ONNX cross-encoder rerank of the top 20

Status: accepted

## Context
The hybrid ranking puts the right passage in the top 20 far more often than at rank 1 (dev slice: Hit@10 0.927). A cross-encoder reads the query and passage together, which a bi-encoder cannot, so it is the standard fix for ordering within a short list. Model: cross-encoder/ms-marco-MiniLM-L6-v2 (apache-2.0, trained on MS MARCO passage ranking; the card does not list Natural Questions, which I checked on the card only). Exported to ONNX (opset 17) and served with ONNX Runtime on CPU.

## Result (reduced dev corpus, 3,452 queries, INT8 model; results/dev_rerank_lora_easy.json)
| | nDCG@10 | MRR@10 | Hit@10 |
|---|---|---|---|
| hybrid (BM25 + dense, RRF) | 0.7831 [0.7722, 0.7933] | 0.7543 | 0.9267 |
| rerank top 10 | 0.8589 [0.8491, 0.8677] | 0.8550 | 0.9267 |
| rerank top 20 | 0.8771 [0.8687, 0.8850] | 0.8674 | 0.9528 |

Paired nDCG@10 gain over hybrid: +0.0758 [0.0691, 0.0822] at depth 10 and +0.0941 [0.0860, 0.1017] at depth 20. Reranking only the top 10 cannot change Hit@10, which is why that column stays at 0.9267.

## Cost (results/dev_rerank_latency.json, 150 queries, 2 vCPU, one query at a time)
Rerank stage p50 / p95: 1,103 / 1,230 ms in fp32 and 797 / 905 ms with INT8 dynamic quantization (91 MB to 23 MB). The full 3,452-query run above used the INT8 model only.

To see what quantization costs in quality, I re-exported both models with scripts/export_reranker.py and scored the first 500 queries at depth 20 (results/dev_rerank_int8_vs_fp32.json). nDCG@10 was 0.8610 [0.8358, 0.8840] for fp32 and 0.8597 [0.8343, 0.8828] for INT8, against 0.7639 [0.7347, 0.7920] for the hybrid ranking. Gains over hybrid were +0.0971 [+0.0763, +0.1192] for fp32 and +0.0958 [+0.0752, +0.1175] for INT8. INT8 sits 0.0013 below fp32 on this sample. I did not run a paired test between the two, so I read the quantization cost as small and untested for significance. INT8 logits differ from fp32 by up to 0.43 on a 12-pair spot check, so quality is judged on rankings. INT8 p50 was 813 ms on the 500 queries. The fp32 pass over the same 500 queries ran at p50 2,293 ms because the host was noisy; a rerun on 150 queries with nothing else running gave 1,123 ms, in line with the earlier figure, and the table above keeps that one.

## Decision
Rerank the top 20 with the INT8 model by default; make depth a setting. 800 ms is too slow for an interactive path on two cores, so a real deployment wants more cores or a GPU, and I have not measured either. The cache absorbs repeated queries but not new ones.

## Caveats
The dev slice is far easier than full NQ (BM25 nDCG@10 is 0.68 there against 0.31 on the full corpus), so the size of the gain will differ on the full corpus, where I have not run the reranker. The LoRA encoder and the cross-encoder both start from MS MARCO training. That does not leak the test queries, but the two stages share a data source and their gains are not independent evidence.

A first run of this evaluation scored one depth twice per query, which produced a nonsense nDCG. I discarded it, fixed the script, and added an assertion of one ranking per query.
