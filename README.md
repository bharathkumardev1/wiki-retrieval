# wiki-retrieval

A passage search service over Wikipedia text (BM25, a LoRA-tuned bi-encoder, compressed vector search, a cross-encoder reranker) with a small agent that answers with a cited passage or refuses.

**The problem.** Retrieval demos show a query and a plausible list of results. This repo asks what each stage of a search stack buys on a corpus large enough for approximate search to matter (2,681,468 passages from BEIR Natural Questions), and what it costs in latency and memory.

**What it does.** BM25 and a MiniLM bi-encoder with a LoRA adapter are fused by reciprocal rank. FAISS IVF-PQ picks candidates and exact float32 vectors re-score them. An INT8 ONNX cross-encoder can rerank the head. FastAPI serves it with a semantic cache and Prometheus metrics, and a LangGraph agent drafts a cited answer, verifies it, and refuses when a relevance score is low.

**Headline result.** On the full corpus (3,452 test queries) the LoRA bi-encoder reaches nDCG@10 0.402 [0.390, 0.414] against 0.290 [0.278, 0.302] for BM25 with Lucene defaults. On a smaller dev slice that I could run on a CPU container, adding the cross-encoder to the hybrid ranking raises nDCG@10 from 0.783 to 0.877 (paired gain +0.094 [+0.086, +0.102]) at a cost of about 800 ms per query on 2 vCPU. Compressed search over the full corpus answers in 2.6 to 4.6 ms against 336 ms for exact search, with a quality loss measured below.

The LoRA adapter's gain over the untuned encoder is +0.040 [+0.033, +0.047] on the dev slice, where I ran a paired test. On the full corpus it is 0.389 to 0.402. The two 95% intervals overlap, which on its own does not show the difference is insignificant, because both arms score the same queries. [PENDING: a paired test on per-query scores is running and this sentence gets replaced with its result.]

**Demo.** No hosted demo. The repo is private and the service runs locally in two commands (see "Run it"). A captured session:

```
$ python -m wikiret.serve &            # dev slice, 104,201 passages, INT8 reranker on the top 30
$ curl -s localhost:8000/v1/search -H 'content-type: application/json' \
    -d '{"query":"who wrote the declaration of independence","k":3}'
{"cache": "miss", "index_version": "59e7ae60418ff582", "results": [
  {"id": "doc849162", "title": "History of the United States (1776\u201389)", "text": "On July 2, 1776, the Second Continental Congress, still meet..."},
  {"id": "doc4861",   "title": "United States Declaration of Independence", "text": "The official copy of the Declaration of Independence was the..."},
  {"id": "doc79950",  "title": "Independence Day (United States)", "text": "During the American Revolution, the legal separation of the ..."}],
 "timings_ms": {"bm25": 2.04, "dense": 4.01, "fuse": 0.23}}
```

That is the first request after start on a shared 2 vCPU container: 1.56 s wall. `timings_ms` covers retrieval only, so the rest is the query encoder, the rerank of 30 candidates and first-request warm-up. The same query again is a cache hit and took 32 ms. Text is cut to 60 characters here and the service returns up to 400.

The corpus licence on BEIR NQ is contradictory (the Hugging Face cards say both cc-by-sa-4.0 and cc-by-nc-sa-3.0), so I treat it as non-commercial and ship scripts and no data. `docs/data.md` lists every source.

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

The IVF-PQ rows have no exact re-scoring, so they show what the compressed codes alone give up. Searching with nlist 4096 and nprobe 64 loses 0.031 nDCG@10 at 96 bytes per vector and 0.069 at 48 bytes. The service re-scores candidates with exact float32 vectors, and I measured that only on the dev slice (below).

<!-- table:full_corpus_cost -->
| Full corpus search stage (4-CPU Kaggle host) | p50 | Vectors held for search |
|---|---|---|
| Exact float32 (flat inner product), one query | 336 ms | 4.12 GB |
| IVF-PQ 48 B/vector, nprobe 64, one query | 2.63 ms | 129 MB of codes, no refine |
| IVF-PQ 96 B/vector, nprobe 64, one query | 4.61 ms | 257 MB of codes, no refine |
<!-- /table:full_corpus_cost -->

The timings are single-query medians on a 4-CPU Kaggle host, the same host for all three rows. Two things to keep in mind when reading the speed and size columns. First, the IVF-PQ rows are the unrefined index at nprobe 64. The configuration that recovers the quality (nprobe 256, 1,000 candidates, exact refine) was measured on the dev slice only, where its dense stage takes 1.26 ms p50, and I have not timed it on the full corpus. Second, refinement needs the float32 vectors in memory, so a served instance holds the 4.12 GB matrix as well as the codes. IVF-PQ makes search faster. It does not make the process smaller.

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

Compressed search with exact re-scoring costs 0.0025 nDCG@10 against exact hybrid search. The paired interval excludes zero, so I report it as a cost. The reranker is the largest single gain in the repo (+0.094 on this slice) and it has the worst latency.

<!-- table:latency -->
| Stage latency on 2 vCPU, one query at a time (ms) | p50 | p95 |
|---|---|---|
| BM25 | 0.7 | 1.27 |
| Dense (IVF-PQ + refine) | 1.26 | 1.89 |
| RRF fuse | 0.1 | 0.2 |
| Cross-encoder, 20 candidates, INT8 | 797 | 905 |
| Cross-encoder, 20 candidates, fp32 | 1103 | 1230 |
<!-- /table:latency -->

Latency is per stage on 2 vCPU, one query at a time, with no other job running for the cross-encoder rows. A 20-candidate rerank at about 800 ms is too slow for every request without a GPU or a shorter candidate list. Reranking the top 10 instead of the top 20 gives 0.859 nDCG@10 rather than 0.877 (`results/dev_rerank_lora_easy.json`). The service reranks whenever a reranker is loaded, 30 candidates by default (`rerank_depth`), so its latency is above the 20-candidate row. Setting `WIKIRET_RERANKER_DIR` to empty turns it off. Whole-service load numbers are under "What is still rough".

### LoRA and hard negatives

<!-- table:lora_ablation -->
| Dev slice, dense retrieval only, paired over queries | nDCG@10 difference [95% CI] |
|---|---|
| LoRA, random negatives, minus untuned | +0.0399 [+0.0328, +0.0468] |
| LoRA, BM25-mined hard negatives, minus untuned | +0.0364 [+0.0297, +0.0432] |
| LoRA, BM25-mined hard negatives, minus random negatives | -0.0035 [-0.0066, -0.0005] |
| LoRA, dense-mined hard negatives, minus random negatives | -0.0023 [-0.0056, +0.0010] |
<!-- /table:lora_ablation -->

I expected mined hard negatives to help. They did not. Random in-batch negatives beat BM25-mined negatives by a small margin whose interval excludes zero, and dense-mined negatives are indistinguishable from random. The shipped adapter uses random negatives. ADR 003 has the setup (rank 16, alpha 32, query and value projections, InfoNCE temperature 0.05, 8,000 pairs from the Natural Questions training split, one epoch) and the open question: training loss sat near 0.04 from step 20, so the batches may be too easy for any negative to matter, and I did not test that. The setup is small, so the result may change at larger scale. The 100,231 training pairs and their source are described in `docs/data.md`; 5 of the 3,452 test queries share their text with a training question, which that note discusses.

### Cited answers and refusal

The agent retrieves, drafts an answer, checks every cited sentence against the cited passage with a deterministic verifier, and refuses when a cross-encoder relevance score for the question and cited passage falls under a threshold.

<!-- table:agent -->
| Agent smoke test, dev corpus, BM25 top 5, extractive drafter | Value |
|---|---|
| Questions | 300 |
| Answered / refused | 251 / 49 |
| Cited passage labelled relevant, among answered | 69.7% |
| Answered with evidence from a different question | 0.7% |
| Injection trials / answers changed | 400 / 0 |
<!-- /table:agent -->

The gate is thresholded on a sheet of 300 answers (200 from the system, 100 controls) that I labelled by hand, twice, as supported, partial or not supported. My two passes agree 84% of the time (kappa 0.73). One person labelled and that person is also the author, so read the rates as an engineering check and not as a benchmark. The labels are in `results/labels/answerability_labels.csv` as ids and labels, and `scripts/answerability_gate.py` regenerates the table from them and the dev slice.

<!-- table:gate -->
| Answerability gate, labelled sheet | Value |
|---|---|
| Labelled system answers | 200 |
| Labels (supported / partial / not_supported) | 48 / 76 / 76 |
| Not-supported answers the gate refuses, cross-fitted | 75.3% (sd 0.095) |
| Acceptable answers it wrongly refuses, cross-fitted | 20.8% (sd 0.073) |
<!-- /table:gate -->

The cross-fitted rates come from 40 random splits, each fitting the threshold on a random half of the sheet and scoring it on the rest. The deployed threshold (4.786) is fitted on all 200 rows, so its in-sample rates (81.6% and 20.2%) are optimistic and I do not quote them. The gate scores how relevant the cited passage is to the question. It does not check that the answer follows from the passage, so a wrong sentence taken from a relevant passage passes. A lexical-overlap judge I tried first had kappa -0.006 against my labels, which is why the verifier checks citations structurally and the quality signal comes from the cross-encoder. ADR 008 covers it.

The drafter in the smoke test is an extractive baseline that picks sentences from the retrieved passages. The local 0.5B model I tried was too weak to write usable answers, so the 69.7% citation-relevance figure measures retrieval and the gate. It says nothing about a language model's writing. The agent takes any generator, and I have not evaluated a real LLM drafter.

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

- **Evaluation data** (001). BEIR NQ for labelled queries at scale, with a 104,201 passage slice of it for iteration. The Wikipedia dumps exercise the Go ingester.
- **Base model** (002). `msmarco-MiniLM-L6-cos-v5`. The better-scoring `all-MiniLM-L6-v2` lists Natural Questions in its training data, which contaminates it for this evaluation, so I measured it once and kept the file to show the gap. The chosen model encodes 2.7 million passages in about 13 minutes on a T4 (fp16) and serves on CPU.
- **LoRA and negatives** (003). LoRA on a frozen encoder with random negatives, since mined ones lost.
- **BM25** (004). My own numpy implementation (Lucene idf, uint8 term frequencies) because `bm25s` ran out of memory on the full corpus. Its default parameters land just below the published BEIR figures (0.329 multifield, 0.306 flat), so I call it the same ballpark and claim no reproduction. Tuned to k1 0.9, b 0.4 on dev queries and scored on held-out queries: 0.309 [0.295, 0.322] against 0.295 for the defaults on the same queries.
- **IVF-PQ with exact refine** (005). Compressed codes pick 1,000 candidates and exact float32 vectors re-score them. The dev-slice setting is nlist 1024, 48 bytes per vector, nprobe 256, chosen on the dev slice and not re-tuned for the full corpus.
- **Verify and refuse** (006), **ONNX reranker** (007), **answerability gate** (008).

## Run it

The retrieval data is not in the repo. Python 3.11 or later:

```
pip install -e '.[serve,agent,dev]'
pytest -q                      # 80 tests, no data needed
```

`python scripts/fetch_nq.py data/nq` downloads BEIR NQ (about 765 MB) from the Hugging Face Hub as corpus.parquet, queries.parquet and qrels_test.tsv. I checked that its output is byte-identical to the files my numbers came from. `python scripts/make_train_pairs.py` fetches the 100,231 training pairs and checks their hash. Then `python scripts/make_dev_corpus.py data/nq data/dev` builds the slice and `scripts/run_ablation.sh` trains and evaluates the adapters (hours on 2 vCPU). `scripts/eval_index.py`, `scripts/eval_rerank.py` and `scripts/answerability_gate.py` reproduce the tables above. The README tables for the full corpus come from the notebooks in `kaggle/` on a T4 and take a few hours.

Training logs to MLflow: add the `train` extra (`pip install -e '.[train]'`).

To serve, with `data/dev` and `artifacts/` built:

```
make run          # docker compose up --build, then http://localhost:8000/healthz
curl -s localhost:8000/v1/search -H 'content-type: application/json' -d '{"query":"who wrote the declaration of independence","k":5}'
```

`python -m wikiret.serve` starts the same service without Docker, and its environment variables are listed at the top of `src/wikiret/serve.py`. `docs/infra-status.md` says which deployment pieces I ran and which I only wrote.

To regenerate or verify the README tables: `python docs/readme_tables.py --write` and `--check`.

## What is still rough

- **Full-corpus adapter gain has no paired test yet.** A Kaggle run to save per-query scores finished once and only its summary numbers survived, so I reran it. [PENDING: replace this bullet with the paired result when `results/` has it.]
- **The refine step and the reranker are measured on the dev slice only.** I have not run either against the full 2.7 million passages.
- **Service load numbers are synthetic and from a 2 vCPU container.** One request at a time, with the load generator sharing the same two CPUs (`results/loadtest.json`). Without the semantic cache, retrieval alone answers at p50 16.5 to 18.4 ms and p95 about 21 to 22 ms (about 54 to 59 requests per second). With the INT8 reranker on the top 20 it answers at p50 871 to 916 ms, p95 about 1,000 ms and p99 1,076 to 1,120 ms (about 1.1 requests per second). With the cache on, a fresh start and 30% generated repeats, p50 was 750 to 824 ms and p99 1.0 to 1.3 s, with 38 to 44 of 100 requests served from the cache. I did not measure concurrency above 1, a real traffic mix or a bigger machine.
- **The Kubernetes manifests have never been applied.** `kind create cluster` fails in my cloud sandbox with a nested-container error. The manifests pass schema validation and nothing more. Terraform was applied and destroyed against LocalStack only (4 resources), and GitHub Actions has not run the workflow.
- **Labels come from one person,** so the gate's threshold and the rates in its table reflect my own reading.
- **Answers are extractive.** No LLM drafter has been evaluated.
- **Latency is CPU-only.** A 20-candidate rerank is about 800 ms on 2 vCPU.
- **Five test queries have the same text as training questions.** I did not rerun the evaluation without them (see `docs/data.md`).

## Licence and data

Code is mine. No licence file is attached yet because the repo is private. BEIR NQ is treated as non-commercial. Wikipedia dumps are GFDL and CC BY-SA 4.0.
