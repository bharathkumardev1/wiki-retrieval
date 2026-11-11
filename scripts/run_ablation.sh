#!/usr/bin/env bash
# Untuned baseline, then the two training arms, each evaluated on the reduced dev corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/dev_eval.py data/nq data/dev --tag untuned
for arm in lora_easy lora_hard; do
  python scripts/train_biencoder.py "configs/$arm.json"
  python scripts/dev_eval.py data/nq data/dev --adapter "artifacts/$arm" --tag "$arm"
done
