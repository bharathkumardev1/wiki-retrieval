"""Agreement between the automatic judge and human labels."""
from __future__ import annotations

import numpy as np

LABELS = ("supported", "partial", "not_supported")


def confusion(human: list[str], judge: list[str], labels: tuple[str, ...] = LABELS) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for h, j in zip(human, judge):
        m[idx[h], idx[j]] += 1
    return m


def kappa(human: list[str], judge: list[str], labels: tuple[str, ...] = LABELS, weights: str | None = None) -> float:
    """Cohen's kappa. `weights="linear"` treats the labels as ordered (supported > partial > not_supported),
    so confusing 'supported' with 'partial' costs less than confusing it with 'not_supported'."""
    m = confusion(human, judge, labels).astype(float)
    n = m.sum()
    if n == 0:
        raise ValueError("no rows to compare")
    k = len(labels)
    i, j = np.indices((k, k))
    w = np.abs(i - j) / (k - 1) if weights == "linear" else (i != j).astype(float)
    expected = np.outer(m.sum(1), m.sum(0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0 if (w * m).sum() == 0 else 0.0  # one label used by both raters: kappa is undefined
    return float(1 - (w * m).sum() / denom)


def percent_agreement(human: list[str], judge: list[str]) -> float:
    return float(np.mean([h == j for h, j in zip(human, judge)]))


def bootstrap_kappa_ci(human: list[str], judge: list[str], n: int = 2000, seed: int = 0, **kw) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    h, j = np.array(human), np.array(judge)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(h), len(h))
        vals.append(kappa(list(h[s]), list(j[s]), **kw))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def binary(labels: list[str]) -> list[str]:
    """Collapse to the question the verifier gates on: fully supported or not."""
    return ["supported" if l == "supported" else "not_supported" for l in labels]


def report(human: list[str], judge: list[str]) -> dict:
    bh, bj = binary(human), binary(judge)
    lo, hi = bootstrap_kappa_ci(human, judge)
    blo, bhi = bootstrap_kappa_ci(bh, bj, labels=("supported", "not_supported"))
    return {
        "n": len(human),
        "percent_agreement": round(percent_agreement(human, judge), 4),
        "kappa_3class": round(kappa(human, judge), 4), "kappa_3class_ci95": [round(lo, 4), round(hi, 4)],
        "kappa_3class_linear_weighted": round(kappa(human, judge, weights="linear"), 4),
        "kappa_supported_vs_not": round(kappa(bh, bj, labels=("supported", "not_supported")), 4),
        "kappa_supported_vs_not_ci95": [round(blo, 4), round(bhi, 4)],
        "confusion_rows_human_cols_judge": {"labels": list(LABELS), "matrix": confusion(human, judge).tolist()},
        "human_label_counts": {l: human.count(l) for l in LABELS},
        "judge_label_counts": {l: judge.count(l) for l in LABELS},
    }
