"""Pick the refuse threshold for the cross-encoder answerability gate from the labelled sheet.

The gate refuses when the cross-encoder score of (question, cited passage) is below a threshold.
Reported rates are cross-fitted: the threshold is chosen on one half of the system rows and scored
on the other half. The threshold written to configs/answerability.json is chosen on all 200 system
rows, so it has no held-out estimate of its own beyond the cross-fitted numbers below.
The labels live in results/labels/answerability_labels.csv as ids and one label per row. The question and
the passage text are looked up from the BEIR NQ queries and the dev corpus by id, so the labelled text is
not redistributed.
python scripts/answerability_gate.py results/labels/answerability_labels.csv data/nq data/dev"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.data import load_queries
from wikiret.rerank import OnnxReranker

labels, nq, dev = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [r for r in csv.DictReader(open(labels, encoding="utf-8-sig")) if r["stratum"] == "system"]
qids, qtext = load_queries(nq / "queries.parquet")
query = dict(zip(qids, qtext))
corpus = pq.read_table(dev / "corpus.parquet", columns=["_id", "title", "text"]).to_pydict()
passage = {i: f"{t} {x}" for i, t, x in zip(corpus["_id"], corpus["title"], corpus["text"])}
rr = OnnxReranker("artifacts/reranker_int8")
y = np.array([r["label"] for r in rows])
s = np.array([rr(query[r["query_id"]], [passage[r["passage_id"]]])[0] for r in rows])
bad = y == "not supported"  # the answers the gate should refuse


def best_threshold(score, is_bad):
    """Maximise balanced accuracy: refuse unsupported answers, keep the rest."""
    cand = np.unique(score)
    bal = [((score[is_bad] < t).mean() + (score[~is_bad] >= t).mean()) / 2 for t in cand]
    return float(cand[int(np.argmax(bal))])


def rates(score, is_bad, t):
    return {"refused_of_not_supported": float((score[is_bad] < t).mean()),
            "refused_of_supported_or_partial": float((score[~is_bad] < t).mean())}


res = []
for seed in range(20):  # repeat the random split so the estimate does not hinge on one shuffle
    idx = np.random.default_rng(seed).permutation(len(rows))
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    for a, b in ((0, 1), (1, 0)):
        t = best_threshold(s[halves[a]], bad[halves[a]])
        res.append(rates(s[halves[b]], bad[halves[b]], t))
cf = {k: [round(float(np.mean([r[k] for r in res])), 3), round(float(np.std([r[k] for r in res])), 3)] for k in res[0]}
final = best_threshold(s, bad)
out = {"n_system_rows": len(rows), "n_not_supported": int(bad.sum()),
       "crossfit_mean_and_sd_over_40_splits": cf,
       "final_threshold": round(final, 3),
       "final_threshold_in_sample": {k: round(v, 3) for k, v in rates(s, bad, final).items()},
       "model": "cross-encoder/ms-marco-MiniLM-L6-v2, INT8 ONNX, max_len 256",
       "note": "labels supplied by the project owner, two passes (agreement 84%, kappa 0.73); single annotator"}
Path("configs").mkdir(exist_ok=True)
Path("configs/answerability.json").write_text(json.dumps({"min_answerability": round(final, 3), "model": out["model"]}, indent=2))
Path("results/answerability_gate.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
