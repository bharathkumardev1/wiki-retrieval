"""Download BEIR Natural Questions from the Hugging Face Hub into the layout the other scripts expect:

    <out>/corpus.parquet    columns _id, title, text (2,681,468 passages)
    <out>/queries.parquet   columns _id, text
    <out>/qrels_test.tsv    query-id, corpus-id, score (3,452 judged test queries)

The files are copied byte for byte from the Hub; nothing is converted. The licence tags on the dataset
cards conflict (cc-by-sa-4.0 and cc-by-nc-sa-3.0), so treat the data as non-commercial and do not commit it.

    python scripts/fetch_nq.py data/nq
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = [
    ("BeIR/nq", "corpus/corpus-00000-of-00001.parquet", "corpus.parquet"),
    ("BeIR/nq", "queries/queries-00000-of-00001.parquet", "queries.parquet"),
    ("BeIR/nq-qrels", "test.tsv", "qrels_test.tsv"),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for repo, name, dest in FILES:
        target = out / dest
        if target.exists():
            print(f"{target} exists, skipping")
            continue
        cached = hf_hub_download(repo, name, repo_type="dataset")
        shutil.copyfile(cached, target)
        print(f"{target}  {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
