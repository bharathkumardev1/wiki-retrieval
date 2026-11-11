from __future__ import annotations

import numpy as np


def rrf(rankings: list[np.ndarray], k: int = 60, depth: int = 100) -> np.ndarray:
    """Reciprocal rank fusion over several ranked id arrays (best first)."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, d in enumerate(r[:depth]):
            score[int(d)] = score.get(int(d), 0.0) + 1.0 / (k + rank + 1)
    return np.array(sorted(score, key=lambda d: -score[d]))
