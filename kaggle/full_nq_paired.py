"""Full BEIR NQ, untuned vs LoRA-tuned encoder, with per-query scores saved for paired tests,
plus timed exact search on the CPU so index speedups have a same-host baseline.

Run in a Kaggle notebook with GPU and Internet on, lora-easy attached. Run this after
`!pip uninstall -y torchao` in a separate cell (Kaggle's preinstalled torchao breaks peft).
Writes /kaggle/working/full_nq_paired.json and /kaggle/working/full_nq_per_query.npz."""
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
SMOKE = int(os.environ.get("SMOKE", "0"))
DATA_DIR = os.environ.get("DATA_DIR")
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def find_adapter():
    root = Path("/kaggle/input")
    for cfg in sorted(root.rglob("adapter_config.json")) if root.exists() else []:
        return str(cfg.parent)
    return os.environ.get("ADAPTER")


def path(repo, name, local):
    return str(Path(DATA_DIR) / local) if DATA_DIR else hf_hub_download(repo, name, repo_type="dataset")


adapter = find_adapter()
print("adapter:", adapter, flush=True)
assert adapter, "attach the lora-easy dataset first"

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
qids = [qid_all[i] for i in sel]
qtext = [qtext_all[i] for i in sel]
rel = [qrels[x] for x in qids]
print(f"{len(ids):,} passages, {len(qtext):,} queries", flush=True)

from sentence_transformers import SentenceTransformer


def per_query(top):
    disc = 1.0 / np.log2(np.arange(2, 12))
    nd, hit, rec100, rec10 = [], [], [], []
    for row, r in zip(top, rel):
        docs = [ids[j] for j in row]
        fl = np.array([d in r for d in docs[:10]], dtype=float)
        nd.append((fl * disc[: len(fl)]).sum() / disc[: min(len(r), 10)].sum())
        hit.append(float(fl.any()))
        rec10.append(fl.sum() / len(r))
        rec100.append(len(set(docs[:100]) & r) / len(r))
    return {"ndcg@10": np.array(nd), "hit@10": np.array(hit), "recall@10": np.array(rec10), "recall@100": np.array(rec100)}


def run(tag, adapter_dir):
    model = SentenceTransformer(BASE_MODEL, device=DEV)
    model.max_seq_length = 128
    if adapter_dir:
        from peft import PeftModel
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, adapter_dir).merge_and_unload()
    if DEV == "cuda":
        model.half()
    t0 = time.time()
    P = model.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    enc_s = time.time() - t0
    Qe = model.encode(qtext, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    del model
    torch.cuda.empty_cache()
    index = faiss.IndexFlatIP(P.shape[1])
    index.add(P)
    _, top = index.search(Qe, 100)
    pq_ = per_query(top)
    # same-host exact search timing: one query at a time, all CPU threads FAISS chooses by default
    lat = []
    for i in range(min(100, len(Qe))):
        t1 = time.perf_counter()
        index.search(Qe[i:i + 1], 100)
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat[10:])
    print(tag, {k: round(float(v.mean()), 4) for k, v in pq_.items()}, f"encode {enc_s:.0f}s exact p50 {np.percentile(lat, 50):.1f} ms", flush=True)
    return pq_, {"tag": tag, "encode_seconds": round(enc_s), "exact_flat_ip_single_query_ms": {
        "p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}}


arrays, meta = {}, []
for tag, ad in (("untuned", None), ("lora_easy", adapter)):
    pq_, m = run(tag, ad)
    meta.append(m)
    for k, v in pq_.items():
        arrays[f"{tag}/{k}"] = v
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "full_nq_per_query.npz", qids=np.array(qids), **arrays)
(OUT / "full_nq_paired.json").write_text(json.dumps({
    "note": "FULL CORPUS" if not SMOKE else "SMOKE TEST ONLY", "base_model": BASE_MODEL, "adapter": adapter,
    "n_passages": len(ids), "n_queries": len(qtext), "device_encode": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
    "cpu_count": os.cpu_count(), "faiss": faiss.__version__, "runs": meta}, indent=2))
print(open(OUT / "full_nq_paired.json").read())
