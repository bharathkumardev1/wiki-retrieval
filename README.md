# wiki-retrieval

A passage search service over Natural Questions text from Wikipedia. It fuses BM25 with a LoRA-tuned bi-encoder, searches vectors through a compressed index that exact vectors re-score, and can rerank the head with an INT8 cross-encoder. A small agent on top answers with a cited passage or refuses.

- **Full corpus (2,681,468 passages, 3,452 queries), dense-only exact search:** nDCG@10 0.402 for the LoRA bi-encoder against 0.290 for my BM25 with Lucene defaults.
- **LoRA against the untuned encoder, paired:** +0.014 nDCG@10 [+0.007, +0.021] on the full corpus, permutation p 0.00012. Small and clearly not noise.
- **Dev slice (104,201 passages):** the served hybrid scores 0.783 and the INT8 cross-encoder lifts it to 0.877, about 800 ms per query.
- **Two corpora, never compared:** the dev slice is much easier than the full corpus, and I have not measured the hybrid or the reranker on the full corpus.
- **Scope:** the FastAPI service (`/healthz`, `POST /v1/search`, `/metrics`) handles search, the semantic cache and the rerank. The LangGraph agent runs from `scripts/agent_demo.py` and `scripts/agent_smoke.py` only. The API does not call it and the Docker image does not include it.

Text in the examples and tables comes from Wikipedia (CC BY-SA) by way of BEIR Natural Questions.

**Demo.** Recorded walkthrough (about 3 minutes): https://www.youtube.com/watch?v=5vtW4iaUe3A. The live search in the video uses the untuned encoder. There is no hosted demo, the repo is private, and the service runs locally (see "Reproducing the tables and serving it"). This session came from the LoRA adapter I trained; the untuned quickstart under "Reproducing the tables and serving it" returns the same top results for this query. A captured session:

```
$ python -m wikiret.serve &            # dev slice, 104,201 passages, LoRA adapter, INT8 reranker on the top 20
$ curl -s localhost:8000/v1/search -H 'content-type: application/json' \
    -d '{"query":"who wrote the declaration of independence","k":3}'
{"cache": "miss", "index_version": "59e7ae60418ff582", "results": [
  {"id": "doc849162", "title": "History of the United States (1776\u201389)", "text": "On July 2, 1776, the Second Continental Congress, still meet..."},
  {"id": "doc4861",   "title": "United States Declaration of Independence", "text": "The official copy of the Declaration of Independence was the..."},
  {"id": "doc79950",  "title": "Independence Day (United States)", "text": "During the American Revolution, the legal separation of the ..."}],
 "timings_ms": {"bm25": 2.04, "dense": 4.01, "fuse": 0.23}}
```

That is the first request after start on a shared 2 vCPU container: 1.56 s wall. `timings_ms` covers retrieval only, so the rest is the query encoder, the rerank of 20 candidates and first-request warm-up. The same query again is a cache hit and took 32 ms. Text is cut to 60 characters here and the service returns up to 400.

The corpus licence on BEIR NQ is contradictory (the Hugging Face cards say both cc-by-sa-4.0 and cc-by-nc-sa-3.0), so I treat it as non-commercial and ship scripts and no data. `docs/data.md` lists every source.

### Headline numbers in detail

All numbers below are component results, each on the corpus named. BM25 reaches 0.290 [0.278, 0.302] with Lucene defaults and 0.309 [0.295, 0.322] tuned, scored on the 2,952 queries not used for tuning (defaults score 0.295 on those same queries). The untuned encoder scores 0.389 on the full corpus. On the dev slice the served hybrid scores 0.783 and the reranker adds +0.094 [+0.086, +0.102], paired. Unrefined IVF-PQ over the full corpus answers in 2.6 to 4.6 ms (nprobe 64, no exact re-score) and gives up 0.03 to 0.07 nDCG@10 against exact search, which takes about 321 ms p50 on that host.

The LoRA gain is +0.040 [+0.033, +0.047] nDCG@10 on the dev slice and +0.014 on the full corpus, both from paired tests. The recall@100 gain on the full corpus is larger, +0.043 [+0.034, +0.052]. The untuned score in the paired run is 0.3883 and the 0.389 above comes from an earlier run of the same encoder. The tuned arm was picked from three trained on the same test queries, so the p-values are not adjusted for that choice. The shipped `results/gpu_full_nq_paired_t4.json` holds per-query nDCG@10 arrays only, so the recall@100 p-value (1e-05) cannot be recomputed from the repo; the Kaggle script now writes every per-query metric for future runs.

## Measured results

Two corpora appear below and they are never compared. The full corpus is the one comparable to published BEIR numbers. The dev slice is every judged passage plus 100,000 random distractors (104,201 passages), which is much easier: tuned BM25 scores 0.68 there (k1 0.9 and b 0.4 were chosen on 500 of the 3,452 queries, which favours BM25 slightly) against 0.309 tuned and 0.290 with defaults on the full corpus. I ran the full corpus on a Kaggle T4 for encoding. The dev slice runs on a 2 vCPU container, which is where the paired tests and latency numbers come from.

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

Paired comparison of the two exact-search arms, per-query scores in `results/gpu_full_nq_paired_t4.json`:

<!-- table:full_corpus_paired -->
| Full corpus, exact search, 3,452 queries | Untuned | LoRA | Paired difference [95% bootstrap CI] | Permutation p | Queries better / worse / tied |
|---|---|---|---|---|---|
| ndcg@10 | 0.3883 | 0.4020 | +0.01367 [+0.00655, +0.02082] | 0.00012 | 783 / 699 / 1970 |
| recall@100 | 0.7863 | 0.8290 | +0.04268 [+0.03389, +0.05185] | 1e-05 | 259 / 86 / 3107 |
<!-- /table:full_corpus_paired -->

The IVF-PQ rows have no exact re-scoring, so they show what the compressed codes alone give up. Searching with nlist 4096 and nprobe 64 loses 0.031 nDCG@10 at 96 bytes per vector and 0.069 at 48 bytes. The service re-scores candidates with exact float32 vectors, and I measured that only on the dev slice (below).

<!-- table:full_corpus_cost -->
| Full corpus search stage (4-CPU Kaggle host) | p50 | Vectors held for search |
|---|---|---|
| Exact float32 (flat inner product), one query | 321 ms (p95 330 ms) | 4.12 GB |
| IVF-PQ 48 B/vector, nprobe 64, one query | 2.63 ms | 129 MB of codes, no refine |
| IVF-PQ 96 B/vector, nprobe 64, one query | 4.61 ms | 257 MB of codes, no refine |
<!-- /table:full_corpus_cost -->

The IVF-PQ timings are single-query medians on a 4-CPU Kaggle host. The exact-search row was timed on that same host by `kaggle/full_nq_paired.py` (p50 322.95 ms untuned and 321.16 ms tuned; the table shows the tuned arm). The IVF-PQ rows are the unrefined index at nprobe 64. The configuration that recovers the quality (nprobe 256, 1,000 candidates, exact refine) was measured on the dev slice, where its dense stage takes 1.26 ms p50, and it is not timed on the full corpus. Compression speeds search and does not shrink a served process: refinement needs the float32 vectors, so an instance holds the 4.12 GB matrix next to the codes.

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

Compressed search with exact re-scoring costs 0.0025 nDCG@10 against exact hybrid search. The paired interval excludes zero, so I report it as a cost. The reranker gives the largest gain on this slice, +0.094, and costs about 800 ms per query.

<!-- table:latency -->
| Stage latency on 2 vCPU, one query at a time (ms) | p50 | p95 | p99 |
|---|---|---|---|
| BM25 | 0.7 | 1.27 |  |
| Dense (IVF-PQ + refine) | 1.26 | 1.89 |  |
| RRF fuse | 0.1 | 0.2 |  |
| Cross-encoder, 20 candidates, INT8, 150 queries | 797 | 905 | 936 |
| Cross-encoder, 20 candidates, INT8, all 3,452 queries | 794 | 1848 | 2007 |
| Cross-encoder, 20 candidates, fp32, 150 queries | 1103 | 1230 | 1338 |
<!-- /table:latency -->

Latency is per stage on 2 vCPU, one query at a time. The 150-query rows ran with no other job going. The all-queries INT8 row has the same median and a much longer tail (p95 1,848 ms, p99 2,007 ms), and I do not know how much of that is host noise; plan around it. A 20-candidate rerank at about 800 ms is too slow for every request without a GPU or a shorter candidate list. Reranking the top 10 gives 0.859 nDCG@10 against 0.877 for the top 20 (`results/dev_rerank_lora_easy.json`). The service reranks the top 20 whenever a reranker is loaded (`WIKIRET_RERANK_DEPTH`), and `WIKIRET_RERANKER_DIR=` with an empty value turns it off. Whole-service load numbers are under "Known gaps".

### LoRA and hard negatives

<!-- table:lora_ablation -->
| Dev slice, dense retrieval only, paired over queries | nDCG@10 difference [95% CI] | hit@10 difference [95% CI] |
|---|---|---|
| LoRA, random negatives, minus untuned | +0.0399 [+0.0328, +0.0468] | +0.0371 [+0.0287, +0.0458] |
| LoRA, BM25-mined hard negatives, minus untuned | +0.0364 [+0.0297, +0.0432] | +0.0336 [+0.0258, +0.0420] |
| LoRA, BM25-mined hard negatives, minus random negatives | -0.0035 [-0.0066, -0.0005] | -0.0035 [-0.0075, +0.0006] |
| LoRA, dense-mined hard negatives, minus random negatives | -0.0023 [-0.0056, +0.0010] | -0.0067 [-0.0116, -0.0020] |
<!-- /table:lora_ablation -->

Mined hard negatives did not beat random in-batch negatives. BM25-mined negatives are behind by a small margin whose nDCG@10 interval excludes zero. Dense-mined negatives are indistinguishable from random on nDCG@10 and significantly behind on hit@10 (-0.0067 [-0.0116, -0.0020]). The shipped adapter uses random negatives. ADR 003 has the setup (rank 16, alpha 32, query and value projections, InfoNCE temperature 0.05, 8,000 pairs from the Natural Questions training split, one epoch). Logged training loss (12 values per run, every 20 steps) averaged 0.063 with random negatives, 0.128 with BM25-mined and 0.180 with dense-mined negatives. I have not tested why mined negatives did not help. Candidates are too few pairs or one epoch, unlabelled positives among the mined passages (NQ judgments are sparse), the rank window the negatives were drawn from, and a single seed; none was run. A larger run may behave differently. The 100,231 training pairs and their source are described in `docs/data.md`, which also covers the 5 of 3,452 test queries that share their text with a training question.

### Cited answers and refusal

The agent retrieves, drafts an answer from the passages, and verifies every cited sentence with a deterministic check: the citation must point at a shown passage, the sentence must be covered by a window of passage sentences, and its numbers and quotes must appear there. A claim that fails is dropped, and the agent reformulates once and then refuses. `agent_demo.py` and `agent_smoke.py` also run an answerability gate when `configs/answerability.json` and `artifacts/reranker_int8` exist: the INT8 cross-encoder scores the question and the cited passage, and a claim whose cited passages all score under the threshold fails. An answer is returned only if every claim passes.

<!-- table:agent -->
| Agent smoke test, dev corpus, BM25 top 5, extractive drafter | Lexical verifier only | With answerability gate (threshold 4.786) |
|---|---|---|
| Questions | 300 | 300 |
| Answered / refused | 251 / 49 | 58 / 242 |
| Cited passage labelled relevant, among answered | 69.7% | 94.8% |
| Answered with evidence from a different question | 0.7% | 0.0% |
| Injection trials / answers changed | 400 / 0 | 400 / 0 |
<!-- /table:agent -->

Both columns use the same 300 questions, BM25 top 5 and the extractive drafter. The gate is far stricter inside the agent than its rates on the label sheet suggest. On the sheet it refused 20.8% of acceptable answers, and in the agent it turns 251 answers into 58. The agent applies the threshold to every claim of an answer, and the sheet scored one cited passage per answer, so I expect an answer with several claims to clear the gate much less often. I read that off `verify()` and did not measure claim counts. The threshold was fitted on 200 answers from this same sampled question set (`make_label_sheet.py` and `agent_smoke.py` share a seed and pool), so the gated column is in-sample for the gate. The jump in citation relevance among answered questions (69.7% to 94.8%) is mostly the effect of answering 193 fewer questions. The injection row is 4 attack sentences times 100 questions, run against an extractive drafter that cannot follow instructions, with detector patterns written for those same sentences. It checks the filtering code path and does not show resistance to a language model being steered.

The gate was thresholded on a sheet of 300 answers (200 from the system, 100 controls) that I labelled by hand, twice, as supported, partial or not supported. I recorded 84% agreement between my two passes (kappa 0.73); the first pass is not shipped, so that figure cannot be recomputed here. One person labelled and that person is also the author, so read the rates as an engineering check. The labels are in `results/labels/answerability_labels.csv` as ids and labels, and `scripts/answerability_gate.py` regenerates the table from them and the dev slice.

<!-- table:gate -->
| Answerability gate, labelled sheet | Value |
|---|---|
| Labelled system answers | 200 |
| Labels (supported / partial / not_supported) | 48 / 76 / 76 |
| Not-supported answers the gate refuses, cross-fitted | 75.3% (sd 0.095) |
| Acceptable answers it wrongly refuses, cross-fitted | 20.8% (sd 0.073) |
<!-- /table:gate -->

The cross-fitted rates come from 40 random splits, each fitting the threshold on a random half of the sheet and scoring it on the rest. The deployed threshold (4.786) is fitted on all 200 rows, so its in-sample rates (81.6% and 20.2%) are optimistic and I do not quote them. The gate scores how relevant the cited passage is to the question. A wrong sentence taken from a relevant passage passes it. A lexical-overlap judge I tried first called 194 of the 200 system answers supported, so its kappa of -0.006 against my labels is what a near-constant predictor produces. The cross-encoder's cutoffs were fitted with cross-fitting and the lexical judge's thresholds were set up front and left untuned, so the two are not a like-for-like comparison. I moved the quality signal to the cross-encoder and kept the structural citation checks. ADR 008 covers it.

The drafter in the smoke test copies sentences from the retrieved passages. I tried Qwen2.5-0.5B-Instruct as the drafter on 5 questions with the lexical verifier (`results/agent_demo_qwen.json`): the agent refused all 5, after empty drafts, drafts with no citation or claims the cited passage did not support, at 49 to 84 seconds per question on 2 vCPUs. Five questions say nothing about a rate. The agent takes any generator, and I have not evaluated a real LLM drafter.

## How it is wired

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

## Choices and reasons

Each has an ADR in `docs/adr/` with the alternatives I considered and the numbers behind the choice.

- **Evaluation data** (001). BEIR NQ for labelled queries at scale, with a 104,201 passage slice of it for iteration. The Wikipedia dumps exercise the Go ingester.
- **Base model** (002). `msmarco-MiniLM-L6-cos-v5`. The better-scoring `all-MiniLM-L6-v2` lists Natural Questions in its training data, which contaminates it for this evaluation, so I measured it once and kept the file to show the gap. The chosen model encodes 2.7 million passages on a T4 (fp16) in 804 s to 1,390 s depending on the session, and serves on CPU.
- **LoRA and negatives** (003). LoRA on a frozen encoder with random negatives, since mined ones lost.
- **BM25** (004). My own numpy implementation (Lucene idf, uint8 term frequencies) because `bm25s` ran out of memory on the full corpus. Its default parameters land just below the published BEIR figures (0.329 multifield, 0.306 flat), so I call it the same ballpark and claim no reproduction. Tuned to k1 0.9, b 0.4 on dev queries and scored on held-out queries: 0.309 [0.295, 0.322] against 0.295 for the defaults on the same queries.
- **IVF-PQ with exact refine** (005). Compressed codes pick 1,000 candidates and exact float32 vectors re-score them. The dev-slice setting is nlist 1024, 48 bytes per vector, nprobe 256, chosen on the dev slice and not re-tuned for the full corpus.
- **Verify and refuse** (006), **ONNX reranker** (007), **answerability gate** (008).

## Reproducing the tables and serving it

The workflow in `.github/workflows/ci.yml` runs the Python tests on 3.11 and 3.13, and GitHub Actions run #5, on the push of commit b349589, showed python 3.11, python 3.13, go, image and terraform all green (ruff checks syntax errors, pyflakes, import order, pycodestyle E4/E7 and bugbear). On a machine without a GPU, install the CPU build of torch first, because `sentence-transformers` otherwise pulls the multi-gigabyte CUDA build:

```
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[serve,agent,dev]'
.venv/bin/python -m pytest -q          # 82 tests, no data needed
```

Serving a fresh clone takes two make targets, with the untuned encoder because the LoRA adapter has to be trained:

```
make dev-data        # downloads BEIR NQ (about 765 MB), builds the 104,201-passage slice, encodes it, exports the reranker
make serve-untuned   # http://localhost:8000, POST /v1/search
```

On a Mac, use Python 3.11 or later (the system Python 3.9 is too old). `make serve-untuned` sets `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` there, because torch and faiss each bring an OpenMP runtime and the server otherwise aborts or the first search hangs (seen on an Apple M2); if you start the server some other way on a Mac, set the same two variables.

`dev-data` is the slow one: the encode step alone took 1,562 to 1,748 s (26 to 29 minutes) per arm on the 2 vCPU container (`encode_seconds` in `results/dev_*.json`; 1,748 s for the untuned encoder that `make serve-untuned` uses). It runs `scripts/fetch_nq.py` (the files are byte-identical to the ones my numbers came from), `scripts/make_dev_corpus.py`, `scripts/embed_dev.py` and `scripts/export_reranker.py`, which also writes the INT8 model the service loads. I ran the export and the server from a clean clone with the slice and embeddings in place; the first two steps I ran in an earlier clean-clone check. To serve the adapter behind the tables, train it (`scripts/train_biencoder.py configs/lora_easy.json`, 8,000 pairs; the MLflow run durations on the 2 vCPU container were 608 s for random negatives, 1,139 s for BM25-mined and 1,204 s for dense-mined, and the re-encode afterwards took 26 to 29 minutes per arm), run `scripts/embed_dev.py data/dev --tag lora_easy --adapter artifacts/lora_easy`, and start `python -m wikiret.serve` without the overrides. `scripts/make_train_pairs.py` fetches the training pairs and `scripts/run_ablation.sh` trains and evaluates every arm. `scripts/eval_index.py`, `scripts/eval_rerank.py` and `scripts/answerability_gate.py` reproduce the dev-slice tables. The full-corpus table comes from the notebooks in `kaggle/` on a T4 and takes a few hours. Training logs to MLflow when you add the `train` extra.

`python -m wikiret.serve` listens on 127.0.0.1 by default; the Docker image sets `WIKIRET_HOST=0.0.0.0` so the published port works. `make run` builds the Docker image and serves the same data (`docker compose`, http://localhost:8000/healthz); it expects `data/dev` with `emb_lora_easy.npy` and `artifacts/` in place, so it follows the training path above. The service's environment variables are listed at the top of `src/wikiret/serve.py`. `docs/infra-status.md` says which deployment pieces I ran and which I only wrote.

To regenerate or verify the README tables: `python docs/readme_tables.py --write` and `--check`.

## Known gaps

- **The full-corpus adapter gain is small, 0.014 nDCG@10.** The paired test puts it above noise (see the detail under the headline numbers).
- **Refine and rerank are measured on the dev slice only,** and the full-corpus hybrid has no number at all.
- **The gate over-refuses inside the agent.** It answers 58 of 300 questions (see the agent table in "Cited answers and refusal").
- **Service load numbers are synthetic.** They come from a 2 vCPU container, one request at a time, with the load generator sharing the same two CPUs (`results/loadtest.json`). Without the semantic cache, retrieval alone answers at p50 16.5 to 18.4 ms and p95 about 21 to 22 ms (about 54 to 59 requests per second). With the INT8 reranker on the top 20 it answers at p50 871 to 916 ms, p95 about 1,000 ms and p99 1,076 to 1,120 ms (about 1.1 requests per second). With the cache on, a fresh start and 30% generated repeats, p50 was 750 to 824 ms and p99 1.0 to 1.3 s, with 38 to 44 of 100 requests served from the cache. Nothing was measured above concurrency 1, on a real traffic mix or on a bigger machine.
- **The Kubernetes manifests have never been applied.** `kind create cluster` fails in my cloud sandbox with a nested-container error, so they have only passed schema validation. Terraform was applied and destroyed against LocalStack only (4 resources). The GitHub Actions jobs (python on 3.11 and 3.13, go, image build, terraform) were green in run #5 on the push of commit b349589. Later commits have not been checked.
- **Labels come from one person,** who is also the author.
- **Answers are extractive,** and no LLM drafter has been evaluated.
- **Inference is CPU-only.** A 20-candidate rerank is about 800 ms on 2 vCPU.
- **Five test queries have the same text as training questions.** The evaluation was not rerun without them (see `docs/data.md`).

## Licence and data

Code is mine. No licence file is attached yet because the repo is private. BEIR NQ is treated as non-commercial. Wikipedia dumps are GFDL and CC BY-SA 4.0.
