"""Effect of the ONNX cross-encoder on the hybrid ranking (reduced dev corpus).

Scores the top DEPTH hybrid candidates once per query; shallower depths reuse those scores.
python scripts/eval_rerank.py data/nq data/dev --tag lora_easy --depth 30"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.data import load_qrels, load_queries
from wikiret.index import HybridIndex
from wikiret.metrics import bootstrap_ci, paired_diff_ci, per_query
from wikiret.rerank import OnnxReranker

ap = argparse.ArgumentParser()
ap.add_argument("nq", type=Path)
ap.add_argument("dev", type=Path)
ap.add_argument("--tag", default="lora_easy")
ap.add_argument("--depth", type=int, default=30)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--reranker", type=Path, default=Path("artifacts/reranker"))
a = ap.parse_args()

t = pq.read_table(a.dev / "corpus.parquet")
ids, titles, texts = t["_id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist()
emb = np.load(a.dev / f"emb_{a.tag}.npy").astype(np.float32)
qids, qtext = load_queries(a.nq / "queries.parquet")
qrels = load_qrels(a.nq / "qrels_test.tsv")
sel = [i for i, q in enumerate(qids) if q in qrels]
if a.limit:
    sel = sel[: a.limit]
qtext = [qtext[i] for i in sel]
rel = [qrels[qids[i]] for i in sel]
Q = np.load(a.dev / f"q_{a.tag}.npy")[: len(sel)] if not a.limit else np.load(a.dev / f"q_{a.tag}.npy")[sel]

idx = HybridIndex(ids, titles, texts, emb, nlist=1024, pq_bytes=48, model_tag=a.tag)
rr = OnnxReranker(a.reranker)
orders = {"hybrid": [], **{f"rerank_top{d}": [] for d in (10, 20, a.depth) if d <= a.depth}}
lat = []
for q, v in zip(qtext, Q):
    order = idx.search(q, v).order
    head = order[: a.depth]
    t0 = time.perf_counter()
    s = rr(q, [f"{titles[i]} {texts[i]}" for i in head])
    lat.append((time.perf_counter() - t0) * 1000)
    orders["hybrid"].append(order)
    for d in (10, 20, a.depth):
        if d <= a.depth:
            h = head[:d]
            orders[f"rerank_top{d}"].append(np.concatenate([h[np.argsort(-s[:d], kind="stable")], order[d:]]))

per, out = {}, {}
for name, o in orders.items():
    ranked = [[ids[j] for j in row] for row in o]
    m10 = per_query(ranked, rel, 10)
    per[name] = m10["ndcg@10"]
    out[name] = {k: {"mean": round(float(v.mean()), 4), "ci95": [round(x, 4) for x in bootstrap_ci(v)]} for k, v in m10.items()}
diffs = {}
for name in orders:
    if name != "hybrid":
        m, lo, hi = paired_diff_ci(per[name], per["hybrid"])
        diffs[f"ndcg@10 {name} minus hybrid"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
lat = np.array(lat)
report = {"note": "REDUCED CORPUS; not comparable to full-corpus numbers", "n_queries": len(qtext), "depth": a.depth,
          "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2 (ONNX, max_len 256)", "metrics": out, "paired_differences": diffs,
          "rerank_ms_per_query": {"p50": round(float(np.percentile(lat, 50))), "p95": round(float(np.percentile(lat, 95))),
                                  "p99": round(float(np.percentile(lat, 99)))},
          "host": "2 vCPU container, one query at a time, ONNX Runtime CPU"}
if not a.limit:
    Path(f"results/dev_rerank_{a.tag}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=1))
