# 001. Evaluate on BEIR Natural Questions, develop on a Simple English slice

Status: accepted

## Context
The project needs queries with human relevance labels and a corpus big enough that approximate search is a real trade-off. Wikipedia dumps are GFDL and CC BY-SA 4.0. The Wikipedia clickstream is CC0 but has no query text, so it cannot supply labelled queries. BEIR NQ has 2,681,468 passages and 3,452 test queries, and published BM25 numbers exist to check against.

The licence on BEIR NQ is unclear. The Hugging Face card for BeIR/nq says cc-by-sa-4.0 and the mteb/nq card says cc-by-nc-sa-3.0. I treat it as non-commercial. The repo ships download scripts, not data, and stays private until that is settled.

## Decision
- Final numbers come from the full BEIR NQ corpus and all test queries.
- Iteration (ablations, training runs) uses a 104,201 passage slice built from the Simple English dump, which my Go ingester processes in under three minutes.
- Slice numbers are never compared with full-corpus numbers. The slice is far easier: BM25 nDCG@10 is 0.68 there and 0.31 on the full corpus.

## Consequences
Every results file says which corpus it ran on. Two sets of numbers exist on purpose and the README keeps them apart.
