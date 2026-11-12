# Case study: building passage search I could defend number by number

Most retrieval demos show one query and a plausible list of results. I wanted to know which parts of a search stack earn their cost, so I built one on a corpus big enough that approximate search is a real trade-off (2,681,468 Wikipedia passages from BEIR Natural Questions) and measured each stage against the one before it. Every number in the README comes from a file in `results/`, and `docs/readme_tables.py --check` fails if the README drifts from those files.

## The stack

A Go ingester turns a Wikipedia dump into passages with deterministic ids. Retrieval is BM25 (my own numpy implementation, because `bm25s` ran out of memory at about 120 million postings) fused by reciprocal rank with a LoRA-tuned MiniLM bi-encoder. The dense side uses FAISS IVF-PQ to pick 1,000 candidates and then re-scores them with the exact float32 vectors. An optional INT8 ONNX cross-encoder reranks the head. A FastAPI service wraps it with a version-keyed semantic cache and Prometheus metrics. On top sits a LangGraph agent that drafts a cited answer, verifies each cited sentence, and refuses when a relevance score for the question and passage is low.

## Compression needs the re-score

On the full corpus, IVF-PQ at 96 bytes per vector runs in 4.6 ms against 336 ms for exact search and takes 257 MB of codes against 4.12 GB of vectors, but it drops nDCG@10 from 0.402 to 0.371. On the 104,201-passage dev slice, adding the exact re-score step recovers all but 0.0035 of the dense nDCG@10 and 0.0025 of the hybrid nDCG@10, with paired intervals that exclude zero in both cases. The compression has a cost and I report it as one. The re-score also needs the float vectors in memory, so the speedup comes with no memory saving for a served process. I have measured the re-score step only on the dev slice.

## Hard negatives lost

The usual advice is to mine hard negatives for contrastive fine-tuning. I tried BM25-mined and dense-mined negatives against plain in-batch negatives, and plain won or tied (BM25-mined: -0.0035 [-0.0066, -0.0005]). I kept the simpler arm and wrote down that I don't know why. The training loss was near 0.04 within 20 steps, so the batches may have been too easy for any negative to matter.

## The judge I replaced

My first answer-quality judge was a lexical-overlap check on whether a cited passage supports an answer. I found out it was useless only after labelling data: I labelled 300 answers by hand over two passes (84% agreement between the passes, kappa 0.73), and the judge's kappa against my labels came out at -0.006. A cross-encoder scoring the question and passage did better (AUC 0.87 for not-supported against the rest), so I rebuilt that stage as a refusal gate. Its threshold comes from all 200 labelled system rows, and the quoted rates come from 40 random half-splits: it refuses 75.3% of answers I labelled not supported and wrongly refuses 20.8% of acceptable ones. A gate that refuses a fifth of good answers suits a cautious assistant and would be wrong for a product that must always answer. It also scores relevance only. A wrong sentence from a relevant passage gets through.

## Mistakes along the way

An evaluation script scored the same ranking twice for one depth setting, and the reranker came out at nDCG@10 0.0003. That number was obviously wrong. I fixed the bug and added an assertion that each query has exactly one ranking, because a milder version of the same bug would have passed unnoticed.

My first plan for testing the full-corpus adapter gain (0.389 to 0.402) was a Kaggle run that saves per-query scores, so a paired bootstrap could settle it. That run finished and I recovered only its summary numbers. I reran it with a script that prints its full result at the end, and until that result is in the README the gain is described as unsettled.

I also wrote in an early ADR that the dev slice came from the Simple English dump. It comes from BEIR NQ itself. A review caught the mismatch and ADR 001 now says so.

## Next

The refine step and the reranker at full-corpus scale come first, since both claims rest on the dev slice. After that, a real language model as the drafter with the gate re-measured against it, a second annotator on the label sheet, and the Kubernetes manifests applied on a working cluster with throughput measured on an idle machine. The README's "What is still rough" section lists the rest.
