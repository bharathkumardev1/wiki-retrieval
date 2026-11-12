"""Encode the dev corpus and write data/dev/emb_<tag>.npy, which `python -m wikiret.serve` loads.

Same text, batch size and fp16 storage as dev_eval.py, without the evaluation. About 30 minutes on 2 vCPU.

python scripts/embed_dev.py data/dev --tag untuned
python scripts/embed_dev.py data/dev --tag lora_easy --adapter artifacts/lora_easy
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("dev", type=Path)
ap.add_argument("--model", default="sentence-transformers/msmarco-MiniLM-L6-cos-v5")
ap.add_argument("--adapter", type=Path, default=None)
ap.add_argument("--tag", default="untuned")
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--limit", type=int, default=0, help="encode only the first N passages (for a quick check)")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

t = pq.read_table(a.dev / "corpus.parquet")
texts = [f"{x} {y}" for x, y in zip(t["title"].to_pylist(), t["text"].to_pylist())]
if a.limit:
    texts = texts[: a.limit]
model = SentenceTransformer(a.model, device="cpu")
model.max_seq_length = a.max_len
if a.adapter:
    from peft import PeftModel

    model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, str(a.adapter)).merge_and_unload()
P = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
out = a.out or a.dev / f"emb_{a.tag}.npy"
np.save(out, P.astype(np.float16))
print(f"wrote {out} {P.shape}")
