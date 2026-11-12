# Case study: what each stage of a passage search stack buys

I wanted to know which stages of a search stack earn their cost, so I built one on a corpus large enough that approximate search is a real trade-off (2,681,468 Wikipedia passages from BEIR Natural Questions) and measured every stage against the one before it. The numbers in the README come from files in `results/`, and `docs/readme_tables.py --check` fails when the README drifts from them.

## The stack

A Go ingester turns a Wikipedia dump into passages with deterministic ids. Retrieval is BM25 (my own numpy implementation, because `bm25s` ran out of memory at about 120 million postings) fused by reciprocal rank with a LoRA-tuned MiniLM bi-encoder. FAISS IVF-PQ picks 1,000 dense candidates and exact float32 vectors re-score them. An optional INT8 ONNX cross-encoder reranks the head. A FastAPI service adds a version-keyed semantic cache and Prometheus metrics. A LangGraph agent drafts a cited answer, verifies each cited sentence, and refuses when verification fails.

## Compression costs quality until the re-score

On the full corpus, IVF-PQ at 96 bytes per vector answers in 4.6 ms and holds 257 MB of codes against 4.12 GB of float vectors, and nDCG@10 falls from 0.402 to 0.371. On the 104,201-passage dev slice, adding the exact re-score step brings dense nDCG@10 back to within 0.0035 of exact search and the hybrid to within 0.0025, with paired intervals that exclude zero in both cases. I report those as costs. The re-score needs the float vectors in memory, so a served process holds the full matrix next to the codes. The refined configuration is timed only on the slice. Exact flat search over the full corpus takes 321 ms p50 on a 4-CPU host (`results/gpu_full_nq_paired_t4.json`), against 2.6 to 4.6 ms for the unrefined index.

## Mined negatives lost

Hard negatives are the standard advice for contrastive fine-tuning. BM25-mined negatives came out behind random in-batch negatives (-0.0035 nDCG@10 [-0.0066, -0.0005]), and dense-mined negatives tied on nDCG@10 and fell behind on hit@10 (-0.0067 [-0.0116, -0.0020]). I kept the simple arm. Logged loss averaged 0.063 with random negatives, 0.128 with BM25-mined and 0.180 with dense-mined negatives. I did not test why mined negatives did not help: too few pairs, unlabelled positives among the mined passages, the sampling window and a single seed are all possible.

## A lexical judge, then a cross-encoder gate

My first answer-quality check was lexical overlap between a cited sentence and its passage. I labelled 300 answers by hand in two passes (I recorded 84% agreement between the passes, kappa 0.73; the first pass is not shipped) and compared. The judge called 194 of the 200 system answers supported, so its kappa against my labels, -0.006, is what a constant predictor scores. A cross-encoder scoring the question and the passage separated better (AUC 0.87 for not-supported against the rest), and I rebuilt that stage as a refusal gate. Cross-fitted over 40 random splits, it refuses 75.3% of the answers I labelled not supported and wrongly refuses 20.8% of acceptable ones. It scores relevance, so a wrong sentence from a relevant passage goes through.

Wiring the gate into the agent showed a second problem. On 300 sampled questions the lexical verifier answers 251. With the gate on, the agent answers 58. The label sheet scored one cited passage per answer and the agent requires every claim of an answer to clear the threshold, which explains a stricter result, but I never counted claims per answer. The threshold was fitted on the same sampled questions, so that run is in-sample. A gate whose sheet rates say 21% false refusals and whose agent refuses 80% needs a threshold fitted per claim before anyone should lean on its numbers.

## Errors I found in my own work

An evaluation script scored the same ranking twice for one depth setting and the reranker came out at nDCG@10 0.0003. That value could not be right. The fix is an assertion that every query has exactly one ranking, so a quieter version of the bug fails loudly.

I first wrote in ADR 001 that the dev slice came from the Simple English dump. It is a slice of BEIR NQ, and the ADR now says so.

The full-corpus adapter gain needed per-query scores for a paired test. The paired run writes per-query scores, and the shipped JSON holds the nDCG@10 arrays. The paired gain is +0.014 nDCG@10 [+0.007, +0.021] and +0.043 recall@100: small in nDCG, clear in the test, with a p-value that does not account for picking the tuned arm among three on the same queries.

## Open questions and next runs

Refine and rerank at full-corpus scale come first, since both claims rest on the slice. After that I want a per-claim gate threshold and a second annotator on the label sheet. The drafter should become a real language model with the gate measured against it. The Kubernetes manifests still need a cluster that works. The README section "Known gaps" lists the rest.
