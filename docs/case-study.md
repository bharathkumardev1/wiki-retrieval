# Case study: building passage search I could defend number by number

## The problem I set myself

Most retrieval demos show a query and a plausible list of results. I wanted to know which parts of a search stack earn their cost, so I built one on a corpus big enough that approximate search is a real trade-off (2,681,468 Wikipedia passages from BEIR Natural Questions) and measured each stage against the one before it. Every number in the README comes from a file in `results/`, and `docs/readme_tables.py --check` fails if the README drifts from those files.

## What I built

A Go ingester turns a Wikipedia dump into passages with deterministic ids. Retrieval is BM25 (my own numpy implementation, because `bm25s` ran out of memory at about 120 million postings) fused by reciprocal rank with a LoRA-tuned MiniLM bi-encoder. The dense side uses FAISS IVF-PQ to pick 1,000 candidates and then re-scores them with the exact float32 vectors. An optional INT8 ONNX cross-encoder reranks the head. A FastAPI service wraps it with a version-keyed semantic cache and Prometheus metrics. On top sits a LangGraph agent that drafts a cited answer, verifies each cited sentence, and refuses when an answerability score is low.

## Three results I would not have guessed

**Compressed search alone is a bad trade, compressed search with re-scoring is a good one.** On the full corpus, IVF-PQ at 96 bytes per vector runs in 4.6 ms against 336 ms for exact search and takes 257 MB against 4.12 GB, but it drops nDCG@10 from 0.402 to 0.371. On the 104,201-passage dev slice, adding the exact re-score step recovers all but 0.0035 of the dense nDCG@10 and 0.0025 of the hybrid nDCG@10, both with confidence intervals that exclude zero. The compression is not free and I report it as a cost. I have only measured the re-score step on the dev slice.

**Hard negatives did not help.** The standard advice is to mine hard negatives for contrastive fine-tuning. I tried BM25-mined and dense-mined negatives against plain in-batch negatives. Plain won or tied (BM25-mined: -0.0035 [-0.0066, -0.0005]). I kept the simpler arm, wrote down that I don't know why (the training loss was near 0.04 within 20 steps, so the batches may have been too easy), and did not claim otherwise.

**My first answer-quality judge was worthless, and I found out because I labelled data.** I built a lexical-overlap judge for whether a cited passage supports an answer. I then labelled 300 answers by hand over two passes (84% self-agreement, kappa 0.73). The judge's kappa against my labels was -0.006. A cross-encoder scoring the question and passage did better (AUC 0.87 for not-supported against the rest), so I rebuilt that stage as a refusal gate with a threshold fitted by cross-validation. It refuses 75.3% of answers I labelled not supported and wrongly refuses 20.8% of acceptable ones. A gate that refuses a fifth of good answers is usable for a cautious assistant and wrong for a product that must always answer; the rates let a reader decide.

## Two things that went wrong

An evaluation script scored the same ranking twice for one depth setting, which produced an nDCG@10 of 0.0003 for the reranker. A number that low is obviously wrong, and I fixed the bug and added an assertion that each query has exactly one ranking. A less dramatic version of the same bug would have passed unnoticed.

A Kaggle run was supposed to save per-query scores so I could test the full-corpus adapter gain with a paired bootstrap. The run finished and I only recovered its summary numbers, so the full-corpus adapter gain (0.389 to 0.402) has overlapping intervals and no paired test. I say that in the README rather than leaning on the dev-slice test.

## What I would do next

Run the refine step and the reranker at full-corpus scale; replace the extractive drafter with a real model and re-measure the gate against it; get a second annotator on the label sheet; apply the Kubernetes manifests to a working cluster and measure throughput on an idle machine. The README's "What is still rough" section lists the rest.
