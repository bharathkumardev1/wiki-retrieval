"""Paired comparison of the dense arms on the reduced dev corpus, from the saved passage embeddings.

python scripts/compare_arms.py data/nq data/dev"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from peft import PeftModel
from sentence_transformers import SentenceTransformer

from wikiret.data import load_qrels, load_queries
from wikiret.metrics import paired_diff_ci, per_query

nq, dev = Path(sys.argv[1]), Path(sys.argv[2])
BASE = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
ids = pq.read_table(dev / "corpus.parquet", columns=["_id"])["_id"].to_pylist()
qids, qtext = load_queries(nq / "queries.parquet")
qrels = load_qrels(nq / "qrels_test.tsv")
sel = [i for i, q in enumerate(qids) if q in qrels]
qtext = [qtext[i] for i in sel]
rel = [qrels[qids[i]] for i in sel]

per = {}
for tag, adapter in (("untuned", None), ("lora_easy", "artifacts/lora_easy"), ("lora_hard", "artifacts/lora_hard"),
                     ("lora_hard_dense", "artifacts/lora_hard_dense")):
    model = SentenceTransformer(BASE, device="cpu")
    model.max_seq_length = 128
    if adapter:
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, adapter).merge_and_unload()
    P = np.load(dev / f"emb_{tag}.npy").astype(np.float32)
    Q = model.encode(qtext, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    tops = []
    for s0 in range(0, len(Q), 256):
        sc = Q[s0:s0 + 256] @ P.T
        part = np.argpartition(-sc, 99, axis=1)[:, :100]
        order = np.argsort(-np.take_along_axis(sc, part, axis=1), axis=1)
        tops.append(np.take_along_axis(part, order, axis=1))
    top = np.vstack(tops)
    m = per_query([[ids[j] for j in row] for row in top], rel, 10)
    per[tag] = m
out = {"note": "REDUCED CORPUS; dense retrieval only; paired bootstrap over queries"}
for metric in ("ndcg@10", "mrr@10", "hit@10"):
    for a, b in (("lora_easy", "untuned"), ("lora_hard", "untuned"), ("lora_hard_dense", "untuned"),
                 ("lora_hard", "lora_easy"), ("lora_hard_dense", "lora_easy"), ("lora_hard_dense", "lora_hard")):
        mean, lo, hi = paired_diff_ci(per[a][metric], per[b][metric])
        out[f"{metric}: {a} minus {b}"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
Path("results/dev_arm_comparison.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
