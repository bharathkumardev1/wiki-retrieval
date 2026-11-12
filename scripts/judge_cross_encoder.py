"""Can the cross-encoder act as the support judge? Scores come from the labelled sheet only.

Thresholds are fitted on one half of the rows and applied to the other half (two-fold cross-fitting),
so every reported prediction is out of sample. The labels are the user-supplied sheet; they are
not touched or created here.
python scripts/judge_cross_encoder.py SHEET.csv JUDGE.csv"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from wikiret.agent.agreement import kappa
from wikiret.rerank import OnnxReranker

AL = {"supported": "supported", "partial": "partial", "not supported": "not_supported"}
LAB = ("supported", "partial", "not_supported")
sheet = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
stratum = {r["row_id"]: r["stratum"] for r in csv.DictReader(open(sys.argv[2], encoding="utf-8-sig"))}
rr = OnnxReranker("artifacts/reranker_int8")
y = np.array([AL[r["human_label"].strip().lower()] for r in sheet])
sys_mask = np.array([stratum[r["row_id"]] == "system" for r in sheet])
# does the extracted sentence answer the question, and does the page it came from answer it
s_ans = np.array([rr(r["query"], [r["system_answer"]])[0] for r in sheet])
s_pas = np.array([rr(r["query"], [f"{r['passage_title']} {r['passage_text']}"])[0] for r in sheet])


def auc(score, pos):
    p, n = score[pos], score[~pos]
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def ordinal_fit(s, labels):
    """Two cutoffs on a score maximising linear-weighted kappa on the fitting rows."""
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    best, arg = -9, None
    for hi in grid:
        for lo in grid[grid < hi]:
            pred = np.where(s >= hi, "supported", np.where(s >= lo, "partial", "not_supported"))
            k = kappa(list(labels), list(pred), LAB, weights="linear")
            if k > best:
                best, arg = k, (lo, hi)
    return arg


def crossfit(score, mask, seed=0):
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    pred = np.empty(len(score), dtype=object)
    for a, b in ((0, 1), (1, 0)):
        lo, hi = ordinal_fit(score[halves[a]], y[halves[a]])
        pred[halves[b]] = np.where(score[halves[b]] >= hi, "supported", np.where(score[halves[b]] >= lo, "partial", "not_supported"))
    return idx, pred


def boot(f, n_rows, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, n_rows, n_rows)) for _ in range(reps)]
    return [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]


out = {"n_rows": len(sheet), "n_system_rows": int(sys_mask.sum()), "label_counts_system": {l: int((y[sys_mask] == l).sum()) for l in LAB}}
for name, score in (("ce_query_answer", s_ans), ("ce_query_passage", s_pas), ("ce_sum", s_ans + s_pas)):
    res = {"auc_supported_vs_rest_system": round(auc(score[sys_mask], y[sys_mask] == "supported"), 3),
           "auc_not_supported_vs_rest_system": round(auc(-score[sys_mask], y[sys_mask] == "not_supported"), 3)}
    idx, pred = crossfit(score, sys_mask)
    h, p = y[idx], pred[idx]
    res["crossfit_system_percent_agreement"] = round(float((h == p).mean()), 3)
    res["crossfit_system_kappa_3class"] = round(kappa(list(h), list(p), LAB), 3)
    res["crossfit_system_kappa_3class_ci95"] = boot(lambda ix: kappa(list(h[ix]), list(p[ix]), LAB), len(h))
    res["crossfit_system_kappa_linear_weighted"] = round(kappa(list(h), list(p), LAB, weights="linear"), 3)
    res["confusion_rows_human_cols_judge"] = [[int(((h == a) & (p == b)).sum()) for b in LAB] for a in LAB]
    out[name] = res
Path("results/judge_cross_encoder.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
