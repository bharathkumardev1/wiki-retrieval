# 004. Own BM25 implementation, checked against published numbers

Status: accepted

## Context
bm25s ran out of memory on the full corpus (about 120M postings), so the only thing I still take from it is the English stopword list, now copied into `src/wikiret/stopwords.py` (33 words) so the results stay the same without the dependency. I wrote a compact numpy implementation: Lucene idf, uint8 term frequencies, one index build of 366 s, 23.4 ms per query single-threaded.

## Check
Default parameters (k1=1.2, b=0.75) on all 3,452 queries give nDCG@10 0.2899 [0.2782, 0.3024]. The BEIR re-evaluation paper (arXiv 2306.07471) reports 0.329 for multifield and 0.306 for flat BM25 on NQ. Mine is below both and the flat figure sits just outside my interval's upper bound at 0.3024. The analyzer and passage boundaries differ from theirs, so I call it the same ballpark and do not claim to have reproduced their number. A weaker BM25 makes the dense and hybrid gains over it look larger than they would against a stronger one, which is why the README gives the published figures next to mine.

## Tuning
k1=0.9, b=0.4 was chosen on 500 dev queries and evaluated on the remaining 2,952: nDCG@10 0.3091 [0.295, 0.3221] against 0.2947 for defaults on the same queries. The dense and hybrid comparisons use the tuned BM25.
