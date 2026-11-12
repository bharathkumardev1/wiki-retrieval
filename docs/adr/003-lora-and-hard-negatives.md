# 003. LoRA fine-tuning of the bi-encoder; hard-negative mining did not help

Status: accepted, with a negative result

## Context
I trained the encoder with LoRA (r=16, alpha=32, query and value projections, in-batch InfoNCE, temperature 0.05) on 8,000 NQ question/passage pairs, one epoch. The arms differ only in negatives: in-batch only, plus one BM25-mined hard negative (ranks 100 to 500) per query, or plus one dense-mined hard negative. All evaluated on the 104,201 passage dev slice with paired bootstrap confidence intervals.

## Results (dense nDCG@10, dev slice)
| Arm | nDCG@10 | Paired diff vs untuned (95% CI) |
|---|---|---|
| untuned | 0.7286 | |
| easy | 0.7686 | +0.0399 [0.0328, 0.0468] |
| BM25 hard | 0.7652 | +0.0364 [0.0297, 0.0432] |
| dense hard | 0.7664 | +0.0376 [0.0302, 0.0449] |

Hard against easy: -0.0035 [-0.0066, -0.0005] for BM25-mined, -0.0023 [-0.0056, +0.0010] for dense-mined. Hybrid (RRF with BM25) scores 0.7856, 0.7859, 0.7823 for the three arms.

## Decision
Keep LoRA with in-batch negatives. At this scale the BM25-mined arm is slightly worse and the dense-mined arm is indistinguishable from easy, so I make no claim that hard negatives help. The training pairs come from `docs/data.md`.

## Open question
Logged loss (12 values per run, every 20 steps) averaged 0.063 with random negatives, 0.128 with BM25-mined and 0.180 with dense-mined negatives. I did not test why mined negatives did not help; candidates are too few pairs or one epoch, unlabelled positives among the mined passages, the sampling window and a single seed. The setup is small (8,000 pairs, one epoch, one negative per query) and the result could change with more data. I also did not correct for the three comparisons.
