# wiki-retrieval

Passage search over Wikipedia text, plus a small agent that answers a question with a cited passage or says it can't. The question this repo tries to answer honestly: on a 2.7 million passage corpus, how much does each stage (BM25, a LoRA-tuned bi-encoder, approximate search, a cross-encoder reranker, a refusal gate) actually buy, and what does it cost in latency and memory?

**Headline.** On the full BEIR Natural Questions corpus (2,681,468 passages, 3,452 test queries), a MiniLM bi-encoder with a LoRA adapter reaches nDCG@10 0.402 [0.390, 0.414] against 0.290 [0.278, 0.302] for BM25 with Lucene defaults. The adapter's gain over the untuned encoder (0.389) is inside the confidence intervals on the full corpus; on the smaller dev slice, where I could run a paired test, it is +0.040 [+0.033, +0.047]. Product-quantised search over the full corpus answers a query in 2.6 to 4.6 ms against 336 ms for exact search, at a quality cost I measured and describe below.

**Demo.** None hosted. The service runs locally (see "Run it"). I plan a private Hugging Face Space over a Simple English Wikipedia slice, and it does not exist yet.

The repo is private. The corpus licence on BEIR NQ is contradictory (the Hugging Face cards say both cc-by-sa-4.0 and cc-by-nc-sa-3.0), so I treat it as non-commercial and ship scripts, not data.

## Results

Two corpora appear below and they are never compared. The full corpus is the one comparable to published BEIR numbers. The dev slice is every judged passage plus 100,000 random distractors (104,201 passages), which is much easier: BM25 scores 0.68 there and 0.31 on the full corpus. I ran the full corpus on a Kaggle T4 for encoding. The dev slice runs on a 2 vCPU container, which is where the paired tests and latency numbers come from.

### Full corpus

<!-- table:full_corpus -->
| Full NQ corpus, 2,681,468 passages, 3,452 queries | nDCG@10 [95% CI] | recall@100 |
|---|---|---|
| BM25, Lucene defaults | 0.290 [0.278, 0.302] | 0.748 |
| Bi-encoder, untuned, exact | 0.389 [0.376, 0.402] | 0.786 |
| Bi-encoder, LoRA, exact | 0.402 [0.390, 0.414] | 0.829 |
| Bi-encoder, LoRA, IVF-PQ 48 B, nprobe 64, no refine | 0.333 [0.320, 0.344] | 0.729 |
| Bi-encoder, LoRA, IVF-PQ 96 B, nprobe 64, no refine | 0.371 [0.358, 0.383] | 0.759 |
<!-- /table:full_corpus -->

The IVF-PQ rows have no exact re-scoring, so they show what the compressed codes alone give up. Searching with nlist 4096 and nprobe 64 loses 0.031 nDCG@10 at 96 bytes per vector and 0.069 at 48 bytes. The service re-scores candidates with exact float32 vectors, which I measured only on the dev slice (below). I have not rerun the refine step at full-corpus scale.

<!-- table:full_corpus_cost -->
| Full corpus search stage (4-CPU Kaggle host) | p50 | Index size |
|---|---|---|
| Exact float32 (flat inner product), one query | 336 ms | 4.12 GB |
| IVF-PQ 48 B/vector, nprobe 64, one query | 2.63 ms | 129 MB |
| IVF-PQ 96 B/vector, nprobe 64, one query | 4.61 ms | 257 MB |
<!-- /table:full_corpus_cost -->

The timings are single-query medians on the 4-CPU Kaggle host, same host for all three rows. Exact search needs the 4.12 GB float32 matrix in memory; the compressed indexes are 129 MB and 257 MB.

### Dev slice: the serving stack

<!-- table:dev_stack -->
| Dev slice, 104,201 passages (not comparable to the table above) | nDCG@10 [95% CI] | Paired difference |
|---|---|---|
| Dense, exact | 0.769 [0.757, 0.780] |  |
| Dense, IVF-PQ + exact refine | 0.765 [0.753, 0.776] | -0.0035 [-0.0053, -0.0019] vs exact |
| BM25 + dense, RRF, exact | 0.786 [0.775, 0.796] |  |
| BM25 + dense, RRF, served path | 0.783 [0.772, 0.793] | -0.0025 [-0.0044, -0.0007] vs exact |
| Served path + INT8 cross-encoder, top 20 | 0.877 [0.869, 0.885] | +0.0941 [+0.0860, +0.1017] vs served path |
<!-- /table:dev_stack -->

Compressed search with exact re-scoring costs 0.0025 nDCG@10 against exact hybrid search, and it is statistically distinguishable from zero, so I report it as a cost instead of calling it free. The reranker is the largest single gain in the repo (+0.094 on this slice), and it also has the worst latency.

<!-- table:latency -->
| Stage latency on 2 vCPU, one query at a time (ms) | p50 | p95 |
|---|---|---|
| BM25 | 0.7 | 1.27 |
| Dense (IVF-PQ + refine) | 1.26 | 1.89 |
| RRF fuse | 0.1 | 0.2 |
| Cross-encoder, 20 candidates, INT8 | 797 | 905 |
| Cross-encoder, 20 candidates, fp32 | 1103 | 1230 |
<!-- /table:latency -->

Latency is end to end per stage on 2 vCPU, one query at a time, with no other job running for the cross-encoder rows. A 20-candidate rerank at about 800 ms is too slow to put on every request without a GPU or a smaller candidate list. Reranking the top 10 instead of the top 20 gives 0.859 nDCG@10 rather than 0.877 (`results/dev_rerank_lora_easy.json`). The service reranks whenever a reranker is loaded, 30 candidates by default (`rerank_depth`), so its latency is higher than the 20-candidate row. Setting `WIKIRET_RERANKER_DIR` to empty turns it off. I have not measured the service end to end.

### LoRA and hard negatives

<!-- table:lora_ablation -->
| Dev slice, dense retrieval only, paired over queries | nDCG@10 difference [95% CI] |
|---|---|
| LoRA, random negatives, minus untuned | +0.0399 [+0.0328, +0.0468] |
| LoRA, BM25-mined hard negatives, minus untuned | +0.0364 [+0.0297, +0.0432] |
| LoRA, BM25-mined hard negatives, minus random negatives | -0.0035 [-0.0066, -0.0005] |
| LoRA, dense-mined hard negatives, minus random negatives | -0.0023 [-0.0056, +0.0010] |
<!-- /table:lora_ablation -->

I expected mined hard negatives to help. They did not. Random in-batch negatives beat BM25-mined negatives by a small, significant margin, and dense-mined negatives are indistinguishable from random. The shipped adapter uses random negatives. ADR 003 has the setup (rank 16, alpha 32, query and value projections, InfoNCE temperature 0.05, 8,000 pairs, one epoch) and what I don't know: training loss sat near 0.04 from step 20, so the batches may simply be too easy for the negatives to matter, and I did not test that. The setup is small (8,000 pairs, one epoch, one negative per query), so the result may not hold at larger scale.

### Cited answers and refusal

The agent retrieves, drafts an answer, checks every cited sentence against the cited passage with a deterministic verifier, and refuses when a cross-encoder answerability score falls under a threshold.

<!-- table:agent -->
| Agent smoke test, dev corpus, BM25 top 5, extractive drafter | Value |
|---|---|
| Questions | 300 |
| Answered / refused | 251 / 49 |
| Cited passage labelled relevant, among answered | 69.7% |
| Answered with evidence from a different question | 0.7% |
| Injection trials / answers changed | 400 / 0 |
<!-- /table:agent -->

The gate is thresholded on a sheet of 300 answers (200 from the system, 100 controls) that I labelled by hand, twice, as supported, partial or not supported. My two passes agree 84% of the time (kappa 0.73). There is one annotator and the annotator is also the author, so read these rates as an engineering check, not a benchmark.

<!-- table:gate -->
| Answerability gate, labelled sheet | Value |
|---|---|
| Labelled system answers | 200 |
| Labels (supported / partial / not_supported) | 48 / 76 / 76 |
| Not-supported answers the gate refuses, cross-fitted | 75.3% (sd 0.095) |
| Acceptable answers it wrongly refuses, cross-fitted | 20.8% (sd 0.073) |
<!-- /table:gate -->

The cross-fitted rates come from 40 random splits, each fitting the threshold on a random half of the sheet and scoring it on the rest. The deployed threshold (4.786) is fitted on all 200 rows, so its in-sample rates (81.6% and 20.2%) are optimistic and I do not quote them. A lexical-overlap judge I tried first had kappa -0.006 against my labels, which is why the verifier checks citations structurally and the quality signal comes from the cross-encoder. ADR 008 covers it.

The drafter in the smoke test is an extractive baseline that picks sentences from the retrieved passages. The local 0.5B model I tried was too weak to write usable answers, so the 69.7% citation-relevance figure measures retrieval and the gate, not a language model's writing. The agent code takes any generator, and I have not evaluated a real LLM drafter.

## Architecture

```
Simple English / enwiki dump (.bz2)
  -> ingest/ (Go): streaming parse, wikitext cleanup, passage split, deterministic ids
  -> passages.jsonl

BEIR NQ corpus + queries + qrels
  -> train_biencoder.py (LoRA, InfoNCE)      -> artifacts/lora_easy
  -> embeddings (fp16 .npy)

query -> BM25 (numpy, Lucene idf)  --------------------\
      -> encoder -> IVF-PQ candidates -> exact rescore --> RRF -> [INT8 cross-encoder] -> results
                                                          ^
                                semantic cache, keyed by index version
```

`src/wikiret/` holds the library: `index.py` (hybrid index and version digest), `cache.py`, `service.py` (FastAPI, `/healthz`, `POST /v1/search`, `/metrics`), `rerank.py`, `serve.py` (env-configured entry point) and `agent/` (verifier, answerability gate, LangGraph wiring). `scripts/` holds every evaluation as a command that writes a file into `results/`. `kaggle/` holds the three GPU notebooks' source.

## Decisions

Each has an ADR in `docs/adr/` with the alternatives I considered and the numbers behind the choice.

- **Evaluation data** (001). BEIR NQ for labelled queries at scale; Wikipedia dumps for the ingester. Dev-slice and full-corpus numbers are kept apart in every results file.
- **Base model** (002). `msmarco-MiniLM-L6-cos-v5`. The better-scoring `all-MiniLM-L6-v2` lists Natural Questions in its training data, so it is contaminated for this evaluation; I measured it once and kept the file so the gap is visible. The chosen model encodes 2.7 million passages in about 13 minutes on a T4 (fp16) and serves on CPU.
- **LoRA and negatives** (003). LoRA on a frozen encoder; random negatives, because mined ones lost.
- **BM25** (004). My own numpy implementation (Lucene idf, uint8 term frequencies) because `bm25s` ran out of memory on the full corpus. With default parameters it lands just below the published BEIR figures (0.329 multifield, 0.306 flat), so I call it the same ballpark, not a reproduction. Tuned to k1 0.9, b 0.4 on dev queries and scored on held-out queries: 0.309 [0.295, 0.322] against 0.295 for the defaults on the same queries.
- **IVF-PQ with exact refine** (005). Compressed codes pick 1,000 candidates, exact float32 vectors re-score them. The dev-slice setting is nlist 1024, 48 bytes per vector, nprobe 256. That setting was chosen on the dev slice and has not been re-tuned for the full corpus.
- **Verify and refuse** (006), **ONNX reranker** (007), **answerability gate** (008).

## Run it

The retrieval data is not in the repo. Python 3.11 or later:

```
pip install -e '.[serve,agent,dev]'
pytest -q                      # 79 tests, no data needed
```

`python scripts/fetch_nq.py data/nq` downloads BEIR NQ (about 765 MB) from the Hugging Face Hub as corpus.parquet, queries.parquet and qrels_test.tsv; I checked that its output is byte-identical to the files my numbers came from. Then `python scripts/make_dev_corpus.py data/nq data/dev` builds the slice, `scripts/run_ablation.sh` trains and evaluates the adapters, and `scripts/eval_index.py`, `scripts/eval_rerank.py` and `scripts/answerability_gate.py` reproduce the tables above. Full-corpus rows come from the notebooks in `kaggle/` on a T4 and take a few hours.

Training logs to MLflow: add the `train` extra (`pip install -e '.[train]'`).

To serve, with `data/dev` and `artifacts/` built:

```
make run          # docker compose up --build, then http://localhost:8000/healthz
curl -s localhost:8000/v1/search -H 'content-type: application/json' -d '{"query":"who wrote the declaration of independence","k":5}'
```

`python -m wikiret.serve` starts the same service without Docker; its environment variables are listed at the top of `src/wikiret/serve.py`. `docs/infra-status.md` says which deployment pieces I ran and which I only wrote.

To regenerate or verify the README tables: `python docs/readme_tables.py --write` and `--check`.

## What is still rough

- **The README's full-corpus claim about the adapter has no paired test.** I built a Kaggle run to save per-query scores, and when it finished I did not retrieve the score file, only the summary numbers. The full-corpus adapter gain (0.389 to 0.402) therefore rests on overlapping intervals plus the dev-slice paired result.
- **The refine step and the reranker are measured on the dev slice only.** I have not run either against the full 2.7 million passages.
- **No service throughput number.** The container's two CPUs were shared with long jobs while I built the service, and a timing would have described the contention. `make loadtest` exists and its output is labelled as synthetic load; it has not been run on an idle machine.
- **The Kubernetes manifests have never been applied.** `kind create cluster` fails in my cloud sandbox with a nested-container error. The manifests pass schema validation and nothing more. Terraform was applied and destroyed against LocalStack only (4 resources), and GitHub Actions has not run the workflow.
- **Labels come from one person.** The gate's threshold and the rates in the table depend on my own labelling.
- **Answers are extractive.** No LLM drafter has been evaluated.
- **Latency is CPU-only.** A 20-candidate rerank is about 800 ms on 2 vCPU.
- **No hosted demo.**

## Licence and data

Code is mine; no licence file is attached yet because the repo is private. BEIR NQ is treated as non-commercial. Wikipedia dumps are GFDL and CC BY-SA 4.0.
