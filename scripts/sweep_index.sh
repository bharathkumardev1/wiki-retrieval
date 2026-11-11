#!/usr/bin/env bash
# Operating-point sweep for the serving index (reduced dev corpus). Choose by overlap with exact search, not by labels.
set -e
for cfg in "64 500" "128 1000" "256 1000"; do
  set -- $cfg
  .venv/bin/python scripts/eval_index.py data/nq data/dev --adapter artifacts/lora_easy --tag lora_easy \
    --nprobe $1 --n-candidates $2 --out results/dev_index_lora_easy_np$1_c$2.json > /dev/null
done
