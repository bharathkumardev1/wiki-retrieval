"""Ranking metrics with binary relevance, plus a paired bootstrap over queries."""
from __future__ import annotations

import numpy as np


def per_query(ranked: list[list[str]], qrels: list[set[str]], k: int = 10) -> dict[str, np.ndarray]:
    """One value per query for each metric. A query with no relevant docs is a
    data error and raises, rather than being scored zero and dragging means."""
    hit, rr, ndcg, rec = [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for docs, rel in zip(ranked, qrels):
        if not rel:
            raise ValueError("query without relevant documents")
        top = docs[:k]
        flags = np.array([d in rel for d in top], dtype=float)
        hit.append(float(flags.any()))
        pos = np.flatnonzero(flags)
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
        ideal = disc[: min(len(rel), k)].sum()
        ndcg.append(float((flags * disc[: len(flags)]).sum() / ideal))
        rec.append(flags.sum() / len(rel))
    return {f"hit@{k}": np.array(hit), f"mrr@{k}": np.array(rr),
            f"ndcg@{k}": np.array(ndcg), f"recall@{k}": np.array(rec)}


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b with a bootstrap interval, resampling queries jointly."""
    d = a - b
    lo, hi = bootstrap_ci(d, n=n, seed=seed)
    return float(d.mean()), lo, hi
