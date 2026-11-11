"""BM25 with compact postings (int32 doc ids, uint8 term frequencies).

Written because the general-purpose libraries hold every token as a Python
object, which does not fit 2.7M Wikipedia passages in 7 GB. Scoring follows
Lucene: idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

_WORD = re.compile(r"\w+")
_STOP = frozenset(STOPWORDS_EN)
_stem = Stemmer.Stemmer("english")


def analyze(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    return _stem.stemWords(words)


class BM25Builder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self._doc: list[np.ndarray] = []
        self._term: list[np.ndarray] = []
        self._tf: list[np.ndarray] = []
        self._len: list[int] = []
        self.n_docs = 0

    def add(self, texts: Iterable[str]) -> None:
        vocab = self.vocab
        docs, terms, lens = [], [], []
        for text in texts:
            toks = analyze(text)
            ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in toks), dtype=np.int64, count=len(toks))
            u, c = np.unique(ids, return_counts=True)
            docs.append(np.full(len(u), self.n_docs, dtype=np.int32))
            terms.append(u.astype(np.int32))
            self._tf.append(np.minimum(c, 255).astype(np.uint8))
            lens.append(len(toks))
            self.n_docs += 1
        self._doc.append(np.concatenate(docs) if docs else np.empty(0, np.int32))
        self._term.append(np.concatenate(terms) if terms else np.empty(0, np.int32))
        self._len.extend(lens)
        # per-doc tf arrays were appended one by one; merge them per call
        k = len(lens)
        merged = np.concatenate(self._tf[-k:])
        del self._tf[-k:]
        self._tf.append(merged)

    def build(self, k1: float = 1.2, b: float = 0.75) -> "BM25":
        doc = np.concatenate(self._doc)
        term = np.concatenate(self._term)
        tf = np.concatenate(self._tf)
        self._doc, self._term, self._tf = [], [], []
        order = np.argsort(term, kind="stable")
        doc, tf = doc[order], tf[order]
        n_terms = len(self.vocab)
        df = np.bincount(term[order], minlength=n_terms)
        del order, term
        indptr = np.zeros(n_terms + 1, dtype=np.int64)
        np.cumsum(df, out=indptr[1:])
        return BM25(self.vocab, indptr, doc, tf, np.asarray(self._len, dtype=np.float32), k1, b)


class BM25:
    def __init__(self, vocab, indptr, doc, tf, doc_len, k1=1.2, b=0.75):
        self.vocab, self.indptr, self.doc, self.tf = vocab, indptr, doc, tf
        self.doc_len = doc_len
        self.n_docs = len(doc_len)
        self.set_params(k1, b)
        df = np.diff(indptr).astype(np.float64)
        self.idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    def set_params(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self._norm = (k1 * (1 - b + b * self.doc_len / self.doc_len.mean())).astype(np.float32)

    def search(self, query: str, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for t in set(analyze(query)):
            j = self.vocab.get(t)
            if j is None:
                continue
            lo, hi = self.indptr[j], self.indptr[j + 1]
            d = self.doc[lo:hi]
            tf = self.tf[lo:hi].astype(np.float32)
            scores[d] += self.idf[j] * tf * (self.k1 + 1) / (tf + self._norm[d])
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top, scores[top]
