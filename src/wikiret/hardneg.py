from __future__ import annotations

import numpy as np

from .bm25 import BM25Builder


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def mine(queries: list[str], pool: list[str], positives: list[int], lo: int = 100, hi: int = 500,
         seed: int = 0) -> list[int]:
    """One hard negative per query: a random pool passage ranked in [lo, hi) by BM25.

    Ranks 1..lo are skipped because unlabelled true positives crowd the top;
    the pool is other queries' answers, so deeper ranks are safer. Passages
    textually identical to the positive are never returned."""
    bd = BM25Builder()
    bd.add(pool)
    idx = bd.build(k1=0.9, b=0.4)
    rng = np.random.default_rng(seed)
    out = []
    for q, pos in zip(queries, positives):
        cand, _ = idx.search(q, hi)
        cand = [c for c in cand[lo:hi] if c != pos and norm(pool[c]) != norm(pool[pos])]
        out.append(int(rng.choice(cand)) if cand else int(rng.integers(len(pool))))
    return out
