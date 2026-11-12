"""Agreement between the automatic judge and the human labels in the filled-in sheet.

python scripts/judge_agreement.py label-sheet/label_sheet.csv label-sheet/judge_labels.csv

Refuses to run on blank labels. Nothing here fabricates or imputes a human judgment."""
import argparse
import csv
import json
import sys
from pathlib import Path

from wikiret.agent.agreement import report

ALIASES = {"supported": "supported", "partial": "partial", "partially": "partial", "partially supported": "partial",
           "not supported": "not_supported", "not_supported": "not_supported", "unsupported": "not_supported"}


def load(sheet: Path, judge: Path, allow_blank: bool):
    human = {r["row_id"]: r["human_label"].strip().lower().replace("-", " ") for r in csv.DictReader(open(sheet, encoding="utf-8-sig"))}
    jrows = {r["row_id"]: r for r in csv.DictReader(open(judge, encoding="utf-8-sig"))}
    blank = [k for k, v in human.items() if not v]
    bad = {k: v for k, v in human.items() if v and v not in ALIASES}
    if bad:
        sys.exit(f"unrecognised labels (use supported / partial / not supported): {list(bad.items())[:5]}")
    if blank and not allow_blank:
        sys.exit(f"{len(blank)} of {len(human)} rows have no human label yet. Fill them in, or pass --only-labelled to score the rest.")
    keep = [k for k, v in human.items() if v]
    if set(keep) - set(jrows):
        sys.exit("sheet and judge file do not match: row_ids differ")
    return [ALIASES[human[k]] for k in keep], [jrows[k]["judge_label"] for k in keep], [jrows[k] for k in keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("judge", type=Path)
    ap.add_argument("--only-labelled", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/judge_agreement.json"))
    a = ap.parse_args()
    h, j, rows = load(a.sheet, a.judge, a.only_labelled)
    out = {"overall": report(h, j)}
    for name, pick in (("stratum_system", lambda r: r["stratum"] == "system"), ("stratum_control", lambda r: r["stratum"] == "control")):
        idx = [i for i, r in enumerate(rows) if pick(r)]
        if len(idx) >= 20:
            out[name] = report([h[i] for i in idx], [j[i] for i in idx])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
