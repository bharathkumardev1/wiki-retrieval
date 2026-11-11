#!/usr/bin/env bash
# Trains the two arms and evaluates each on the reduced dev corpus.
# The untuned baseline is produced first by: python scripts/dev_eval.py data/nq data/dev --tag untuned
set -euo pipefail
cd "$(dirname "$0")/.."
for arm in lora_easy lora_hard; do
  python scripts/train_biencoder.py "configs/$arm.json"
  python scripts/dev_eval.py data/nq data/dev --adapter "artifacts/$arm" --tag "$arm"
done
