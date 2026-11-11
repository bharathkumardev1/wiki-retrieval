"""BM25 over the BEIR NQ corpus.

python scripts/bm25_baseline.py data/nq

Reports the Lucene-default setting on all test queries, then tunes k1 and b on a
fixed 500-query dev split and reports the tuned setting on the other queries only."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.bm25 import BM25Builder
from wikiret.data import load_qrels, load_queries
from wikiret.metrics import bootstrap_ci, per_query

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--out", type=Path, default=Path("results/bm25.json"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

t0 = time.time()
ids: list[str] = []
bd = BM25Builder()
pf = pq.ParquetFile(a.root / "corpus.parquet")
for batch in pf.iter_batches(batch_size=100_000, columns=["_id", "title", "text"]):
    d = batch.to_pydict()
    ids.extend(d["_id"])
    bd.add(f"{t} {x}" for t, x in zip(d["title"], d["text"]))
    if len(ids) % 500_000 < 100_000:
        print(f"  {len(ids):,} passages, vocab {len(bd.vocab):,}, {time.time()-t0:.0f}s", flush=True)
idx = bd.build()
del bd
print(f"index built: {idx.n_docs:,} passages, {len(idx.vocab):,} terms, {len(idx.doc):,} postings, {time.time()-t0:.0f}s", flush=True)

qids, qtext = load_queries(a.root / "queries.parquet")
qrels = load_qrels(a.root / "qrels_test.tsv")
keep = [i for i, q in enumerate(qids) if q in qrels]
qids, qtext = [qids[i] for i in keep], [qtext[i] for i in keep]
rel = [qrels[q] for q in qids]
rng = np.random.default_rng(a.seed)
perm = rng.permutation(len(qids))
dev, held = perm[:500], perm[500:]


def run(sel):
    t = time.time()
    ranked = [[ids[j] for j in idx.search(qtext[i], 100)[0]] for i in sel]
    ms = (time.time() - t) / len(sel) * 1000
    r = [rel[i] for i in sel]
    return per_query(ranked, r, 10), per_query(ranked, r, 100), ms


def summarize(m10, m100, ms):
    out = {"n_queries": len(m10["hit@10"]), "mean_ms_per_query_1thread": round(ms, 1)}
    for name, v in {**m10, "recall@100": m100["recall@100"]}.items():
        lo, hi = bootstrap_ci(v)
        out[name] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


result = {"n_passages": idx.n_docs, "n_terms": len(idx.vocab)}
idx.set_params(1.2, 0.75)
result["lucene_default_k1_1.2_b_0.75_all_test_queries"] = summarize(*run(list(range(len(qids)))))
print(json.dumps(result, indent=1), flush=True)

grid = []
for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
    for b in (0.4, 0.6, 0.75, 0.9):
        idx.set_params(k1, b)
        m10, _, _ = run(dev)
        grid.append({"k1": k1, "b": b, "ndcg@10_dev": round(float(m10["ndcg@10"].mean()), 4)})
best = max(grid, key=lambda g: g["ndcg@10_dev"])
idx.set_params(best["k1"], best["b"])
result["dev_grid"] = grid
result["tuned"] = {"k1": best["k1"], "b": best["b"], **summarize(*run(held))}
idx.set_params(1.2, 0.75)
result["default_on_same_held_out_queries"] = summarize(*run(held))
a.out.parent.mkdir(exist_ok=True)
a.out.write_text(json.dumps(result, indent=2))
print("done", f"{time.time()-t0:.0f}s", flush=True)
