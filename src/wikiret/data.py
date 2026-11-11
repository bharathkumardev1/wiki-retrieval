from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_corpus(path: Path, limit: int | None = None) -> tuple[list[str], list[str]]:
    """Returns (doc ids, title + text) in file order. The order is the index order."""
    t = pq.read_table(path, columns=["_id", "title", "text"])
    if limit:
        t = t.slice(0, limit)
    ids = t["_id"].to_pylist()
    texts = [f"{a} {b}" for a, b in zip(t["title"].to_pylist(), t["text"].to_pylist())]
    return ids, texts


def load_queries(path: Path) -> tuple[list[str], list[str]]:
    t = pq.read_table(path)
    return t["_id"].to_pylist(), t["text"].to_pylist()


def load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            q, d, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                out[q].add(d)
    return out
