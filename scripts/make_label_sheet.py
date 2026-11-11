"""Build the human labelling sheet for validating the automatic support judge.

Each row asks: does this passage support this answer to this question? The human fills the
`human_label` column. The judge's own labels go to a separate file so the annotator is not
anchored by them.

Two strata, both sampled with a fixed seed:
  system   the first answer sentence the extractive agent drafts from BM25 top-5, paired with the
           passage it cites (natural mix of right and wrong evidence)
  control  the same answer paired with a top passage retrieved for a different, unrelated query
           (should be 'not supported'; kept separate so agreement on easy rows is not mistaken for skill)

python scripts/make_label_sheet.py data/nq data/dev --out label-sheet"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.agent import ExtractiveGenerator, Passage
from wikiret.agent.text import strip_markers
from wikiret.agent.verify import Claim, VerifyConfig, check_claim, judge_label, split_claims
from wikiret.bm25 import BM25Builder
from wikiret.data import load_corpus, load_qrels, load_queries

ap = argparse.ArgumentParser()
ap.add_argument("nq", type=Path)
ap.add_argument("dev", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--system-rows", type=int, default=200)
ap.add_argument("--control-rows", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ids, texts = load_corpus(a.dev / "corpus.parquet")
titles = pq.read_table(a.dev / "corpus.parquet", columns=["title"])["title"].to_pylist()
raw_text = pq.read_table(a.dev / "corpus.parquet", columns=["text"])["text"].to_pylist()
qids, qtext = load_queries(a.nq / "queries.parquet")
qrels = load_qrels(a.nq / "qrels_test.tsv")
idset = set(ids)
# keep queries whose gold passage is in the dev corpus, so the sample includes questions the corpus can answer
queries = [(q, t) for q, t in zip(qids, qtext) if q in qrels and qrels[q] & idset]

bd = BM25Builder()
bd.add(texts)
bm = bd.build(k1=0.9, b=0.4)
gen, cfg = ExtractiveGenerator(), VerifyConfig()


def passage(i: int) -> Passage:
    return Passage(ids[i], titles[i], raw_text[i])


rng = np.random.default_rng(a.seed)
order = rng.permutation(len(queries))
drafted, skipped = [], 0
for qi in order:
    if len(drafted) >= a.system_rows + a.control_rows:
        break
    q, text = queries[qi]
    top = [passage(i) for i in bm.search(text, 5)[0]]
    claims = split_claims(gen.draft(text, top))
    if not claims or not claims[0].citations:
        skipped += 1  # the extractive drafter found nothing to quote; no answer to label
        continue
    cited = top[claims[0].citations[0] - 1]
    drafted.append({"qid": q, "query": text, "claim": claims[0].text, "cited": cited,
                    "top1": top[0], "gold": qrels[q]})

system, pool = drafted[: a.system_rows], drafted[a.system_rows:]
rows = []
for d in system:
    rows.append(("system", d, d["cited"]))
for d in pool[: a.control_rows]:
    # an unrelated query's top passage, never one of this query's gold passages
    others = [o for o in drafted if o["qid"] != d["qid"] and o["top1"].id not in d["gold"]]
    rows.append(("control", d, others[int(rng.integers(len(others)))]["top1"]))

rng.shuffle(rows)
a.out.mkdir(parents=True, exist_ok=True)
sheet, judge = [], []
for stratum, d, p in rows:
    rid = hashlib.sha1(f"{d['qid']}|{p.id}|{stratum}".encode()).hexdigest()[:8]
    check = check_claim(d["query"], Claim(d["claim"], (1,)), [p], cfg)
    sheet.append({"row_id": rid, "query": d["query"], "system_answer": strip_markers(d["claim"]),
                  "passage_title": p.title, "passage_text": p.text, "human_label": "", "notes": ""})
    judge.append({"row_id": rid, "stratum": stratum, "query_id": d["qid"], "passage_id": p.id,
                  "is_gold_passage": int(p.id in d["gold"]), "support": check.support, "relevance": check.relevance,
                  "reject_reason": "" if check.ok else check.reason, "judge_label": judge_label(check, cfg)})
for name, data in (("label_sheet.csv", sheet), ("judge_labels.csv", judge)):
    with open(a.out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
meta = {"seed": a.seed, "rows": len(rows), "system_rows": sum(r[0] == "system" for r in rows),
        "control_rows": sum(r[0] == "control" for r in rows), "queries_skipped_no_draft": skipped,
        "judge_label_counts": {l: sum(j["judge_label"] == l for j in judge) for l in ("supported", "partial", "not_supported")},
        "gold_passage_share_in_system_rows": round(float(np.mean([j["is_gold_passage"] for j in judge if j["stratum"] == "system"])), 3)}
(a.out / "sheet_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
