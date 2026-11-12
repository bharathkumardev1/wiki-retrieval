#!/usr/bin/env bash
# Trains the three adapters and evaluates each, plus the untuned encoder, on the reduced dev corpus.
# Needs data/nq (scripts/fetch_nq.py), data/dev (scripts/make_dev_corpus.py) and data/train/nq_pairs.parquet
# (scripts/make_train_pairs.py). On a 2 vCPU machine the three trainings take hours.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_train_pairs.py data/train/nq_pairs.parquet
python scripts/dev_eval.py data/nq data/dev --tag untuned
for arm in lora_easy lora_hard lora_hard_dense; do
  python scripts/train_biencoder.py "configs/$arm.json"
  python scripts/dev_eval.py data/nq data/dev --adapter "artifacts/$arm" --tag "$arm"
done
python scripts/compare_arms.py data/nq data/dev
