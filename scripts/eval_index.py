"""Measure the serving index on the reduced dev corpus: exact dense, raw IVF-PQ, IVF-PQ with
exact refine, and each fused with BM25. Reduced corpus, so not comparable to full-corpus numbers.

python scripts/eval_index.py data/nq data/dev --adapter artifacts/lora_easy --tag lora_easy"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.data import load_qrels, load_queries
from wikiret.fusion import rrf
from wikiret.index import HybridIndex
from wikiret.metrics import bootstrap_ci, paired_diff_ci, per_query

ap = argparse.ArgumentParser()
ap.add_argument("nq", type=Path)
ap.add_argument("dev", type=Path)
ap.add_argument("--tag", default="lora_easy")
ap.add_argument("--adapter", type=Path, default=None)
ap.add_argument("--model", default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
ap.add_argument("--nlist", type=int, default=1024)
ap.add_argument("--pq-bytes", type=int, default=48)
ap.add_argument("--nprobe", type=int, default=16)
ap.add_argument("--n-candidates", type=int, default=200)
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

t = pq.read_table(a.dev / "corpus.parquet")
ids, titles, texts = t["_id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist()
emb = np.load(a.dev / f"emb_{a.tag}.npy").astype(np.float32)
qids, qtext = load_queries(a.nq / "queries.parquet")
qrels = load_qrels(a.nq / "qrels_test.tsv")
sel = [i for i, q in enumerate(qids) if q in qrels]
qtext = [qtext[i] for i in sel]
rel = [qrels[qids[i]] for i in sel]

qpath = a.dev / f"q_{a.tag}.npy"
if qpath.exists():
    Q = np.load(qpath)
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cpu")
    m.max_seq_length = 128
    if a.adapter:
        from peft import PeftModel
        m[0].auto_model = PeftModel.from_pretrained(m[0].auto_model, a.adapter).merge_and_unload()
    Q = m.encode(qtext, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(qpath, Q)

t0 = time.time()
idx = HybridIndex(ids, titles, texts, emb, nlist=a.nlist, pq_bytes=a.pq_bytes, nprobe=a.nprobe, n_candidates=a.n_candidates, model_tag=a.tag)
build_s = time.time() - t0

rank = {"dense_exact": [], "ivfpq_raw": [], "ivfpq_refine": [], "hybrid_exact": [], "hybrid_raw": [], "hybrid_refine": []}
lat = {"bm25": [], "dense": [], "fuse": []}
for q, v in zip(qtext, Q):
    bm, _ = idx.bm25.search(q, 100)
    exact = idx.search_exact_dense(v, 100)
    _, cand = idx.ivf.search(v[None], idx.n_candidates)
    cand = cand[0][cand[0] >= 0]
    raw = cand[:100]
    ref = cand[np.argsort(-(idx.emb[cand] @ v), kind="stable")][:100]
    rank["dense_exact"].append(exact)
    rank["ivfpq_raw"].append(raw)
    rank["ivfpq_refine"].append(ref)
    rank["hybrid_exact"].append(rrf([exact, bm]))
    rank["hybrid_raw"].append(rrf([raw, bm]))
    res = idx.search(q, v)
    rank["hybrid_refine"].append(res.order)
    for k in lat:
        lat[k].append(res.timings_ms[k])

per, out = {}, {}
for name, r in rank.items():
    ranked = [[ids[j] for j in row] for row in r]
    m10, m100 = per_query(ranked, rel, 10), per_query(ranked, rel, 100)
    per[name] = m10["ndcg@10"]
    row = {}
    for k, vals in {**m10, "recall@100": m100["recall@100"]}.items():
        lo, hi = bootstrap_ci(vals)
        row[k] = {"mean": round(float(vals.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out[name] = row
diffs = {}
for x, y in (("ivfpq_raw", "dense_exact"), ("ivfpq_refine", "dense_exact"), ("ivfpq_refine", "ivfpq_raw"),
             ("hybrid_refine", "hybrid_exact"), ("hybrid_raw", "hybrid_exact")):
    m, lo, hi = paired_diff_ci(per[x], per[y])
    diffs[f"ndcg@10 {x} minus {y}"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
overlap = {n: round(float(np.mean([len(set(r[:100]) & set(e[:100])) / 100 for r, e in zip(rank[n], rank["dense_exact"])])), 4)
           for n in ("ivfpq_raw", "ivfpq_refine")}
report = {"note": "REDUCED CORPUS; not comparable to full-corpus numbers", "adapter": str(a.adapter), "n_passages": len(ids),
          "n_queries": len(qtext), "index": idx.config, "index_version": idx.version, "build_seconds": round(build_s),
          "overlap_at_100_with_exact_dense": overlap, "metrics": out, "paired_differences": diffs,
          "stage_latency_ms_p50_p95": {k: [round(float(np.percentile(v, 50)), 2), round(float(np.percentile(v, 95)), 2)]
                                       for k, v in lat.items()},
          "host": "2 vCPU container, single query at a time"}
path = a.out or Path(f"results/dev_index_{a.tag}.json")
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=1))
