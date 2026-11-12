# 001. Evaluate on BEIR Natural Questions, develop on a Simple English slice

Status: accepted

## Context
The project needs queries with human relevance labels and a corpus big enough that approximate search is a real trade-off. Wikipedia dumps are GFDL and CC BY-SA 4.0. The Wikipedia clickstream is CC0 but has no query text, so it cannot supply labelled queries. BEIR NQ has 2,681,468 passages and 3,452 test queries, and published BM25 numbers exist to check against.

The licence on BEIR NQ is unclear. The Hugging Face card for BeIR/nq says cc-by-sa-4.0 and the mteb/nq card says cc-by-nc-sa-3.0. I treat it as non-commercial. The repo ships download scripts and no data, and stays private until that is settled. `docs/data.md` lists every source and its licence status.

## Decision
Final numbers come from the full BEIR NQ corpus and all 3,452 test queries. Ablations and training runs iterate on a 104,201 passage slice of that same corpus (`scripts/make_dev_corpus.py`): every passage judged relevant to any test query, plus 100,000 seeded random distractors. The Go ingester and the Simple English Wikipedia dump are a separate path. They exercise the ingestion code and have no labelled queries, so no retrieval number comes from them.

The slice is far easier than the full corpus. BM25 nDCG@10 is 0.68 there and 0.31 on the full corpus, so slice numbers are never compared with full-corpus ones. Every results file names the corpus it ran on, and the README keeps the two sets of numbers in separate tables.
