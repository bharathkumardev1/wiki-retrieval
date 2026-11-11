"""BM25, dense and hybrid on the reduced dev corpus (see make_dev_corpus.py).

python scripts/dev_eval.py data/nq data/dev --model sentence-transformers/all-MiniLM-L6-v2"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from wikiret.bm25 import BM25Builder
from wikiret.data import load_qrels, load_queries
from wikiret.fusion import rrf
from wikiret.metrics import bootstrap_ci, paired_diff_ci, per_query

ap = argparse.ArgumentParser()
ap.add_argument("nq", type=Path)
ap.add_argument("dev", type=Path)
ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--adapter", type=Path, default=None)
ap.add_argument("--tag", default="untuned")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
out_path = a.out or Path(f"results/dev_{a.tag}.json")

t = pq.read_table(a.dev / "corpus.parquet")
ids = t["_id"].to_pylist()
texts = [f"{x} {y}" for x, y in zip(t["title"].to_pylist(), t["text"].to_pylist())]
qids, qtext = load_queries(a.nq / "queries.parquet")
qrels = load_qrels(a.nq / "qrels_test.tsv")
sel = [i for i, q in enumerate(qids) if q in qrels]
qids, qtext = [qids[i] for i in sel], [qtext[i] for i in sel]
rel = [qrels[q] for q in qids]

bd = BM25Builder()
bd.add(texts)
bm = bd.build(k1=0.9, b=0.4)
bm_rank = [bm.search(q, 100)[0] for q in qtext]

emb_path = Path(f"data/dev/emb_{a.tag}.npy")
model = SentenceTransformer(a.model, device="cpu")
model.max_seq_length = a.max_len
if a.adapter:
    from peft import PeftModel
    model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, a.adapter).merge_and_unload()
t0 = time.time()
if emb_path.exists():
    P = np.load(emb_path).astype(np.float32)
else:
    P = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.save(emb_path, P.astype(np.float16))
enc_s = time.time() - t0
Q = model.encode(qtext, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
S = Q @ P.T
dense_rank = [np.argsort(-s)[:100] for s in S]
hyb_rank = [rrf([d, b]) for d, b in zip(dense_rank, bm_rank)]


def evaluate(rank):
    r = [[ids[j] for j in row] for row in rank]
    return per_query(r, rel, 10), per_query(r, rel, 100)


res, per = {}, {}
for name, rank in (("bm25_tuned", bm_rank), ("dense", dense_rank), ("hybrid_rrf", hyb_rank)):
    m10, m100 = evaluate(rank)
    per[name] = m10["ndcg@10"]
    row = {}
    for k, v in {**m10, "recall@100": m100["recall@100"]}.items():
        lo, hi = bootstrap_ci(v)
        row[k] = {"mean": round(float(v.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    res[name] = row
diffs = {}
for x, y in (("dense", "bm25_tuned"), ("hybrid_rrf", "bm25_tuned"), ("hybrid_rrf", "dense")):
    m, lo, hi = paired_diff_ci(per[x], per[y])
    diffs[f"ndcg@10 {x} minus {y}"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
out = {"note": "REDUCED CORPUS (relevant passages + seeded distractors); not comparable to full-corpus numbers",
       "model": a.model, "adapter": str(a.adapter) if a.adapter else None, "max_len": a.max_len,
       "n_passages": len(ids), "n_queries": len(qids), "encode_seconds": round(enc_s),
       "metrics": res, "paired_differences": diffs}
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
