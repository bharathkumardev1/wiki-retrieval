"""Semantic cache for search results.

A hit means: a previous query embedded within `threshold` cosine similarity of this one,
answered against the same index version. Entries from another index version are never
returned and are dropped on the next write, so rebuilding the index cannot serve stale
passages."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np


class SemanticCache:
    def __init__(self, threshold: float = 0.97, max_entries: int = 1024) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: OrderedDict[int, tuple[str, np.ndarray, Any]] = OrderedDict()
        self._next = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, qvec: np.ndarray, version: str) -> tuple[Any, float] | None:
        best, best_sim = None, -1.0
        for key, (v, vec, value) in self._entries.items():
            if v != version:
                continue
            sim = float(vec @ qvec)
            if sim > best_sim:
                best, best_sim, best_key = value, sim, key
        if best is not None and best_sim >= self.threshold:
            self._entries.move_to_end(best_key)
            return best, best_sim
        return None

    def put(self, qvec: np.ndarray, version: str, value: Any) -> None:
        for key in [k for k, (v, _, _) in self._entries.items() if v != version]:
            del self._entries[key]
        self._entries[self._next] = (version, qvec.astype(np.float32), value)
        self._next += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
