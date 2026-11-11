"""Full BEIR NQ corpus: exact vs FAISS IVF-PQ dense search, optionally with a LoRA adapter.

Run in a Kaggle notebook with GPU and Internet on (the GPU encodes; FAISS runs on CPU).
Writes /kaggle/working/full_nq_faiss.json.
For a fine-tuned run, attach the adapter folder as notebook input and set ADAPTER below."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "peft", "faiss-cpu"], check=False)

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
ADAPTER = None  # e.g. "/kaggle/input/lora-easy/lora_easy"
SMOKE = int(os.environ.get("SMOKE", "0"))
DATA_DIR = os.environ.get("DATA_DIR")
OUT = Path(os.environ.get("OUT", "/kaggle/working/full_nq_faiss.json"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def path(repo, name, local):
    return str(Path(DATA_DIR) / local) if DATA_DIR else hf_hub_download(repo, name, repo_type="dataset")


t = pq.read_table(path("BeIR/nq", "corpus/corpus-00000-of-00001.parquet", "corpus.parquet"), columns=["_id", "title", "text"])
if SMOKE:
    t = t.slice(0, SMOKE)
ids = t["_id"].to_pylist()
texts = [f"{a} {b}" for a, b in zip(t["title"].to_pylist(), t["text"].to_pylist())]
q = pq.read_table(path("BeIR/nq", "queries/queries-00000-of-00001.parquet", "queries.parquet"))
qrels = {}
with open(path("BeIR/nq-qrels", "test.tsv", "qrels_test.tsv"), encoding="utf-8") as f:
    next(f)
    for line in f:
        a, d, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels.setdefault(a, set()).add(d)
idset = set(ids)
qid_all, qtext_all = q["_id"].to_pylist(), q["text"].to_pylist()
sel = [i for i, x in enumerate(qid_all) if x in qrels and (not SMOKE or qrels[x] & idset)]
qtext = [qtext_all[i] for i in sel]
rel = [qrels[qid_all[i]] for i in sel]
print(f"{len(ids):,} passages, {len(qtext):,} queries", flush=True)


def metrics(rank_rows, k=10):
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    nd, h, rec100 = [], [], []
    for row, r in zip(rank_rows, rel):
        docs = [ids[j] for j in row]
        fl = np.array([d in r for d in docs[:k]], dtype=float)
        nd.append((fl * disc[: len(fl)]).sum() / disc[: min(len(r), k)].sum())
        h.append(float(fl.any()))
        rec100.append(len(set(docs[:100]) & r) / len(r))
    rng = np.random.default_rng(0)

    def s(v):
        v = np.array(v)
        m = v[rng.integers(0, len(v), size=(1000, len(v)))].mean(axis=1)
        return {"mean": round(float(v.mean()), 4), "ci95": [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]}
    return {"ndcg@10": s(nd), "hit@10": s(h), "recall@100": s(rec100)}


from sentence_transformers import SentenceTransformer

model = SentenceTransformer(BASE_MODEL, device=DEV)
model.max_seq_length = 128
if ADAPTER:
    from peft import PeftModel
    model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, ADAPTER).merge_and_unload()
if DEV == "cuda":
    model.half()
t0 = time.time()
P = model.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
enc_s = time.time() - t0
Qe = model.encode(qtext, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
d = P.shape[1]
print(f"encoded {len(P):,} in {enc_s:.0f}s", flush=True)

result = {"note": "FULL CORPUS" if not SMOKE else "SMOKE TEST ONLY", "base_model": BASE_MODEL, "adapter": ADAPTER,
          "n_passages": len(ids), "n_queries": len(qtext), "device_encode": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
          "cpu_count": os.cpu_count(), "faiss": faiss.__version__, "encode_seconds": round(enc_s)}

# exact search (reference ranking)
flat = faiss.IndexFlatIP(d)
flat.add(P)
_, exact_top = flat.search(Qe, 100)
result["exact"] = metrics(exact_top)


def single_query_ms(index, n=200):
    lat = []
    for i in range(min(n, len(Qe))):
        t1 = time.perf_counter()
        index.search(Qe[i:i + 1], 100)
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat[20:])
    return {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2)}


n = len(P)
nlist = min(4096, max(16, n // 40))
rng = np.random.default_rng(0)
train = P[rng.choice(n, size=min(n, 200_000), replace=False)]
result["ivfpq"] = []
for m_sub in (48, 96):
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, m_sub, 8, faiss.METRIC_INNER_PRODUCT)
    t1 = time.time()
    index.train(train)
    index.add(P)
    build_s = time.time() - t1
    sweep = []
    for nprobe in (1, 4, 16, 64):
        index.nprobe = min(nprobe, nlist)
        _, top = index.search(Qe, 100)
        overlap = float(np.mean([len(set(a) & set(b)) / 100 for a, b in zip(top, exact_top)]))
        sweep.append({"nprobe": index.nprobe, "overlap_at_100_with_exact": round(overlap, 4),
                      "metrics": metrics(top), "single_query_ms_cpu": single_query_ms(index)})
        print(f"  PQ{m_sub} nprobe {index.nprobe}: overlap {overlap:.3f}", flush=True)
    result["ivfpq"].append({"nlist": nlist, "pq_bytes_per_vector": m_sub, "build_seconds": round(build_s),
                            "code_bytes_total": int(n * m_sub), "float32_bytes_total": int(n * d * 4), "sweep": sweep})
    del index
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result["exact"], indent=1))
