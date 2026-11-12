"""Run the agent on a few dev questions and show each step. --hf uses a local open-weight model
through transformers (downloads it on first use); the default is the extractive baseline.

The answerability gate is on when configs/answerability.json and artifacts/reranker_int8 exist; --no-gate turns it off.

python scripts/agent_demo.py data/nq data/dev --n 5 [--hf Qwen/Qwen2.5-0.5B-Instruct] [--no-gate]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wikiret.agent import ExtractiveGenerator, Passage, VerifyConfig, build_agent, default_verify_config
from wikiret.bm25 import BM25Builder
from wikiret.data import load_corpus, load_qrels, load_queries

ap = argparse.ArgumentParser()
ap.add_argument("nq", type=Path)
ap.add_argument("dev", type=Path)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--hf", default=None)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-gate", action="store_true")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

ids, texts = load_corpus(a.dev / "corpus.parquet")
t = pq.read_table(a.dev / "corpus.parquet", columns=["title", "text"])
titles, raw = t["title"].to_pylist(), t["text"].to_pylist()
qids, qtext = load_queries(a.nq / "queries.parquet")
qrels = load_qrels(a.nq / "qrels_test.tsv")
pool = [(q, x) for q, x in zip(qids, qtext) if q in qrels and qrels[q] & set(ids)]
sample = [pool[i] for i in np.random.default_rng(a.seed).permutation(len(pool))[: a.n]]
bd = BM25Builder()
bd.add(texts)
bm = bd.build(k1=0.9, b=0.4)


class Bm25:
    def retrieve(self, query, k):
        return [Passage(ids[i], titles[i], raw[i]) for i in bm.search(query, k)[0]]


if a.hf:
    from wikiret.agent.hf_backend import TransformersGenerator
    gen = TransformersGenerator(a.hf)
else:
    gen = ExtractiveGenerator()
cfg = VerifyConfig() if a.no_gate else default_verify_config()[0]
ask = build_agent(Bm25(), gen, cfg=cfg)
rows = []
for q, x in sample:
    t0 = time.time()
    r = ask(x)
    rows.append({"question": x, "answer": r.answer, "refused": r.refused, "cited": r.cited_passage_ids,
                 "gold_cited": any(p in qrels[q] for p in r.cited_passage_ids), "attempts": r.attempts,
                 "seconds": round(time.time() - t0, 1), "verify_rejections": [s.get("rejected") for s in r.trace if s["step"] == "verify"]})
    print(json.dumps(rows[-1], indent=1), flush=True)
if a.out:
    a.out.write_text(json.dumps(rows, indent=2))
