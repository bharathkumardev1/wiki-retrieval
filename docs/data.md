# Data sources and licences

Nothing in this table is committed to the repo. Scripts fetch each source, and the ones that matter for
reproducibility check a hash.

| What | Where it comes from | Fetched by | Licence |
|---|---|---|---|
| BEIR NQ corpus, queries, test qrels (2,681,468 passages, 3,452 judged queries) | `BeIR/nq` and `BeIR/nq-qrels` on the Hugging Face Hub | `scripts/fetch_nq.py` | The cards disagree (cc-by-sa-4.0 on one, cc-by-nc-sa-3.0 on the other). I treat it as non-commercial. |
| Training pairs (100,231 question and answer-passage rows) | `pair` subset of `sentence-transformers/natural-questions` on the Hub, built from the Natural Questions training split | `scripts/make_train_pairs.py` | The dataset card declares none. The Natural Questions card on the Hub declares cc-by-sa-3.0, which I use as the working assumption. |
| Simple English Wikipedia dump | dumps.wikimedia.org | `ingest/` (Go) | GFDL and CC BY-SA 4.0 |

I confirmed that `data/train/nq_pairs.parquet`, the file every adapter in `results/` was trained on, is
identical to the upstream parquet: same 100,231 rows, same bytes (sha256 `8cfb5e5a7cb1...`). The script
pins that hash.

Training and test come from the same Natural Questions release but different splits. `train_biencoder.py`
drops every training pair whose answer text appears in the dev corpus. Matching question text after
lowercasing and stripping punctuation, 5 of the 3,452 test queries (0.14%) also appear among the training
questions. The answer filter removes the training pair for 2 of them; the pairs for the other 3 stay in the
training set (8,000 pairs are sampled from the roughly 100,000, so a given pair is trained on only if it was
drawn). I did not rerun any evaluation with those 5 queries excluded. At 5 of 3,452 queries the effect on
a mean is small, but it is not zero.
