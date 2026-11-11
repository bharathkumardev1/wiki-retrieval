"""Dense retrieval on the FULL BEIR Natural Questions corpus (2,681,468 passages) on one GPU.

Run in a Kaggle notebook with GPU and Internet on. Writes /kaggle/working/full_nq_dense.json.
Optional: attach a LoRA adapter folder as notebook input and set ADAPTERS below.
Metrics here are comparable to published BEIR numbers (full corpus), unlike the reduced dev corpus."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "peft"], check=False)

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
ADAPTERS = {}  # e.g. {"lora_hard": "/kaggle/input/lora-hard"}
SMOKE = int(os.environ.get("SMOKE", "0"))  # >0: use only this many passages, for a quick test
DATA_DIR = os.environ.get("DATA_DIR")  # local copy of the three files; skips the download
OUT = Path(os.environ.get("OUT", "/kaggle/working/full_nq_dense.json"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEV == "cuda"


def fetch(repo, name):
    if DATA_DIR:
        return str(Path(DATA_DIR) / name.split("/")[-1])
    return hf_hub_download(repo, name, repo_type="dataset")


corpus_path = fetch("BeIR/nq", "corpus/corpus-00000-of-00001.parquet") if not DATA_DIR else str(Path(DATA_DIR) / "corpus.parquet")
queries_path = fetch("BeIR/nq", "queries/queries-00000-of-00001.parquet") if not DATA_DIR else str(Path(DATA_DIR) / "queries.parquet")
qrels_path = fetch("BeIR/nq-qrels", "test.tsv") if not DATA_DIR else str(Path(DATA_DIR) / "qrels_test.tsv")

t = pq.read_table(corpus_path, columns=["_id", "title", "text"])
if SMOKE:
    t = t.slice(0, SMOKE)
ids = t["_id"].to_pylist()
texts = [f"{a} {b}" for a, b in zip(t["title"].to_pylist(), t["text"].to_pylist())]
print(f"{len(ids):,} passages", flush=True)
q = pq.read_table(queries_path)
qrels = {}
with open(qrels_path, encoding="utf-8") as f:
    next(f)
    for line in f:
        a, d, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels.setdefault(a, set()).add(d)
id_set = set(ids)
sel = [i for i, x in enumerate(q["_id"].to_pylist())
       if x in qrels and (not SMOKE or qrels[x] & id_set)]
qids = [q["_id"].to_pylist()[i] for i in sel]
qtext = [q["text"].to_pylist()[i] for i in sel]
rel = [qrels[x] for x in qids]
print(f"{len(qids):,} queries", flush=True)


def per_query(ranked, qrels_, k):
    hit, rr, ndcg, rec = [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for docs, r in zip(ranked, qrels_):
        flags = np.array([d in r for d in docs[:k]], dtype=float)
        hit.append(float(flags.any()))
        pos = np.flatnonzero(flags)
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
        ndcg.append(float((flags * disc[: len(flags)]).sum() / disc[: min(len(r), k)].sum()))
        rec.append(flags.sum() / len(r))
    return {f"hit@{k}": np.array(hit), f"mrr@{k}": np.array(rr), f"ndcg@{k}": np.array(ndcg),
            f"recall@{k}": np.array(rec)}


def ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def run(tag, adapter=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BASE_MODEL, device=DEV)
    model.max_seq_length = 128
    if adapter:
        from peft import PeftModel
        model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, adapter).merge_and_unload()
    if HALF:
        model.half()
    torch.manual_seed(0)
    t0 = time.time()
    P = model.encode(texts, batch_size=512, convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=False).to(DEV)
    if DEV == "cuda":
        torch.cuda.synchronize()
    enc_s = time.time() - t0
    print(f"[{tag}] encoded {len(P):,} passages in {enc_s:.0f}s ({len(P)/enc_s:.0f}/s)", flush=True)
    Q = model.encode(qtext, batch_size=256, convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=False).to(DEV)
    top = []
    for s in range(0, len(Q), 128):
        top.append(torch.topk(Q[s:s + 128] @ P.T, k=min(100, len(P)), dim=1).indices.cpu())
    top = torch.cat(top).numpy()
    ranked = [[ids[j] for j in row] for row in top]
    m10, m100 = per_query(ranked, rel, 10), per_query(ranked, rel, 100)
    metrics = {}
    for name, v in {**m10, "recall@100": m100["recall@100"]}.items():
        metrics[name] = {"mean": round(float(v.mean()), 4), "ci95": ci(v)}
    # single-query latency: encode + exact search, batch size 1, after warm-up
    lat = []
    for i in range(min(300, len(qtext))):
        t1 = time.perf_counter()
        e = model.encode([qtext[i]], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        torch.topk(e @ P.T, k=100, dim=1)
        if DEV == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat[20:])
    return {"tag": tag, "adapter": adapter, "encode_seconds": round(enc_s), "passages_per_second": round(len(P) / enc_s),
            "metrics": metrics,
            "single_query_ms_encode_plus_exact_search": {"p50": round(float(np.percentile(lat, 50)), 2),
                                                          "p95": round(float(np.percentile(lat, 95)), 2),
                                                          "p99": round(float(np.percentile(lat, 99)), 2)}}


result = {
    "note": "FULL CORPUS, comparable to published BEIR numbers" if not SMOKE else "SMOKE TEST ONLY: truncated corpus",
    "base_model": BASE_MODEL, "n_passages": len(ids), "n_queries": len(qids),
    "precision": "fp16" if HALF else "fp32", "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
    "torch": torch.__version__, "runs": [],
}
result["runs"].append(run("untuned"))
for tag, path in ADAPTERS.items():
    result["runs"].append(run(tag, path))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=1))
