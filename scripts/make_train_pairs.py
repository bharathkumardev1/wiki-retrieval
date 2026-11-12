"""Fetch the (question, answer passage) pairs the LoRA adapters train on.

    python scripts/make_train_pairs.py data/train/nq_pairs.parquet

The file is the `pair` subset of sentence-transformers/natural-questions on the Hugging Face Hub: 100,231
rows with columns `query` and `answer`, built from the Natural Questions training split. It is copied
byte for byte; nothing is filtered here. train_biencoder.py drops any pair whose answer text also appears
in the dev corpus, so the adapters never see a passage they are scored on.

The copy I trained from has sha256 8cfb5e5a7cb1cd09dbb035a581aa42ce5db5852acb3e70f070d89d7035887de9 and
this script checks for it, so a changed upstream file fails loudly instead of quietly moving the results.
"""

import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "sentence-transformers/natural-questions"
FILE = "pair/train-00000-of-00001.parquet"
SHA256 = "8cfb5e5a7cb1cd09dbb035a581aa42ce5db5852acb3e70f070d89d7035887de9"
ROWS = 100_231


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        shutil.copyfile(hf_hub_download(REPO, FILE, repo_type="dataset"), out)
    digest = sha256(out)
    if digest != SHA256:
        sys.exit(f"{out} has sha256 {digest}, expected {SHA256}. The upstream file changed; "
                 "training on it would not reproduce the reported adapters.")
    import pyarrow.parquet as pq

    n = pq.ParquetFile(out).metadata.num_rows
    if n != ROWS:
        sys.exit(f"{out} has {n} rows, expected {ROWS}")
    print(f"{out}: {n} pairs, sha256 {digest[:12]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train/nq_pairs.parquet"))
