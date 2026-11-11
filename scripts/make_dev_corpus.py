"""Reduced corpus for CPU-scale experiments: every passage judged relevant to any
test query, plus N seeded random distractors from the full BEIR NQ corpus.

Metrics on this corpus are NOT comparable to full-corpus numbers (far fewer
distractors); every result file written from it says so."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wikiret.data import load_qrels

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--distractors", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

qrels = load_qrels(a.root / "qrels_test.tsv")
relevant = set().union(*qrels.values())
t = pq.read_table(a.root / "corpus.parquet")
ids = t["_id"].to_pylist()
is_rel = np.array([i in relevant for i in ids])
pool = np.flatnonzero(~is_rel)
pick = np.random.default_rng(a.seed).choice(pool, size=a.distractors, replace=False)
keep = np.sort(np.concatenate([np.flatnonzero(is_rel), pick]))
a.out.mkdir(parents=True, exist_ok=True)
pq.write_table(t.take(pa.array(keep)), a.out / "corpus.parquet")
print(f"{len(relevant):,} relevant + {a.distractors:,} distractors = {len(keep):,} passages")
