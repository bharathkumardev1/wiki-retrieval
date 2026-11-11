# 004. Own BM25 implementation, checked against published numbers

Status: accepted

## Context
bm25s ran out of memory on the full corpus (about 120M postings). I wrote a compact numpy implementation: Lucene idf, uint8 term frequencies, one index build of 366 s, 23.4 ms per query single-threaded.

## Check
Default parameters (k1=1.2, b=0.75) on all 3,452 queries give nDCG@10 0.2899 [0.2782, 0.3024]. The BEIR re-evaluation paper (arXiv 2306.07471) reports 0.329 for multifield and 0.306 for flat BM25 on NQ. Mine is below both and the flat figure sits just outside my interval's upper bound at 0.3024. The analyzer and passage boundaries differ from theirs, so I treat this as the same ballpark, not a reproduction.

## Tuning
k1=0.9, b=0.4 was chosen on 500 dev queries and evaluated on the remaining 2,952: nDCG@10 0.3091 [0.295, 0.3221] against 0.2947 for defaults on the same queries. The dense and hybrid comparisons use the tuned BM25.
