import math

import numpy as np

from wikiret.bm25 import BM25Builder, analyze

DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "A fox is a small omnivorous mammal",
    "Dogs are domesticated mammals, not natural wild animals",
    "Quantum chromodynamics describes the strong interaction",
]


def reference(query, docs, k1=1.2, b=0.75):
    toks = [analyze(d) for d in docs]
    n, avg = len(docs), sum(len(t) for t in toks) / len(docs)
    scores = []
    for t in toks:
        s = 0.0
        for q in set(analyze(query)):
            df = sum(q in x for x in toks)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = t.count(q)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


def build(chunks):
    bd = BM25Builder()
    for c in chunks:
        bd.add(c)
    return bd.build()


def test_matches_reference_formula():
    idx = build([DOCS])
    for q in ["fox", "mammals dogs", "strong interaction quantum", "nothing matches zzz"]:
        top, sc = idx.search(q, k=4)
        got = np.zeros(4)
        got[top] = sc
        assert np.allclose(got, reference(q, DOCS), atol=1e-5), q


def test_chunked_build_equals_single_build():
    a = build([DOCS])
    b = build([DOCS[:1], DOCS[1:3], DOCS[3:]])
    ta, sa = a.search("fox mammal", k=4)
    tb, sb = b.search("fox mammal", k=4)
    assert list(ta) == list(tb) and np.allclose(sa, sb)


def test_ranking_order_and_k_larger_than_corpus():
    idx = build([DOCS])
    top, sc = idx.search("fox", k=50)
    assert len(top) == 4 and list(sc) == sorted(sc, reverse=True)
    assert set(top[:2]) == {0, 1}


def test_stopword_list_is_the_one_the_results_used():
    from wikiret.stopwords import STOPWORDS_EN

    assert len(STOPWORDS_EN) == 33
    assert {"the", "of", "and", "is"} <= STOPWORDS_EN
