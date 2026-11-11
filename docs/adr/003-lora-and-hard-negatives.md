# 003. LoRA fine-tuning of the bi-encoder; hard-negative mining did not help

Status: accepted, with a negative result

## Context
I trained the encoder with LoRA (r=16, alpha=32, query and value projections, in-batch InfoNCE, temperature 0.05) on 8,000 NQ question/passage pairs, one epoch. Three arms: in-batch negatives only, one BM25-mined hard negative (ranks 100 to 500), one dense-mined hard negative. All evaluated on the 104,201 passage dev slice with paired bootstrap confidence intervals.

## Results (dense nDCG@10, dev slice)
| Arm | nDCG@10 | Paired diff vs untuned (95% CI) |
|---|---|---|
| untuned | 0.7286 | |
| easy | 0.7686 | +0.0399 [0.0328, 0.0468] |
| BM25 hard | 0.7652 | +0.0364 [0.0297, 0.0432] |
| dense hard | 0.7664 | +0.0376 [0.0302, 0.0449] |

Hard against easy: -0.0035 [-0.0066, -0.0005] for BM25-mined, -0.0023 [-0.0056, +0.0010] for dense-mined. Hybrid (RRF with BM25) scores 0.7856, 0.7859, 0.7823 for the three arms.

## Decision
Keep LoRA. Do not claim hard negatives help. At this scale they did not: the BM25-mined arm is slightly worse and the dense-mined arm is indistinguishable from easy.

## What I do not know
Training loss was near 0.04 from step 20, so batches may simply be too easy for the signal to matter. I have not tested that. 8,000 pairs, one epoch and one negative per query is a small setup, and the result may not hold at larger scale. No correction for multiple comparisons was applied.
