"""English stopword list used by the BM25 analyzer.

This is the list shipped by bm25s (MIT licence), copied here so the package does not depend on bm25s for
a constant. It is the same list the reported BM25 numbers were computed with: changing it changes every
BM25 result in results/. tests/test_bm25.py pins the size and a few members."""

STOPWORDS_EN = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it',
    'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they',
    'this', 'to', 'was', 'will', 'with',
})
