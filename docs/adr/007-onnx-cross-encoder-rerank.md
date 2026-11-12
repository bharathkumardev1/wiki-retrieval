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
Rerank stage p50 / p95: 1,103 / 1,230 ms in fp32 and 797 / 905 ms with INT8 dynamic quantization (91 MB to 23 MB). On the same 150 queries nDCG@10 at depth 20 was 0.8804 in fp32 and 0.8845 in INT8, so quantization did not cost quality here; the full 3,452-query run used the INT8 model only.

A second check re-exported both models with scripts/export_reranker.py and scored the first 500 queries (results/dev_rerank_int8_vs_fp32.json): nDCG@10 at depth 20 was 0.8610 in fp32 and 0.8597 in INT8, against 0.7639 for the hybrid ranking. INT8 is 0.0013 lower on this larger sample, which is the reverse sign of the 150-query result, so I read the cost of quantization as roughly zero and have not run a paired test between the two. INT8 logits differ from fp32 by up to 0.43 on a 12-pair spot check, which is why quality is judged on rankings and not on logit parity. INT8 p50 was 813 ms on the 500 queries. The fp32 pass over the same 500 queries ran at p50 2,293 ms because the host was noisy; a rerun on 150 queries with nothing else running gave 1,123 ms, in line with the earlier figure, and the table above keeps that one.

## Decision
Rerank the top 20 with the INT8 model by default; make depth a setting. 800 ms is too slow for an interactive path on two cores, so a real deployment wants more cores or a GPU, and I have not measured either. The cache absorbs repeated queries but not new ones.

## Limits
- Reduced corpus: the dev slice is far easier than full NQ (BM25 nDCG@10 is 0.68 there against 0.31 on the full corpus), so the size of the gain will differ on the full corpus. I have not run the reranker there.
- Both the LoRA encoder and the cross-encoder start from MS MARCO training, so the two stages share a data source. That is not leakage of the test queries, but the gains are not independent evidence.
- A first run of this evaluation had a bookkeeping bug (a depth was scored twice per query). It was discarded and rerun, and the script now asserts one ranking per query.
