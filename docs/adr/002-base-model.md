# 002. Base encoder: msmarco-MiniLM-L6-cos-v5

Status: accepted

## Context
sentence-transformers/all-MiniLM-L6-v2 scores 0.835 dense nDCG@10 on the dev slice, but its model card lists natural_questions in its training data. That makes it contaminated for an NQ evaluation. I measured it anyway and kept the file (results/dev_minilm_all_contaminated.json) so the gap is visible.

## Decision
Use sentence-transformers/msmarco-MiniLM-L6-cos-v5 as the base. Its card does not list NQ. I read the card only; I did not audit the training data.

## Consequences
The untuned dense baseline is lower (0.7286 on the slice) and the fine-tuning gain is measured against a model that has not seen the test task. If the card is wrong, the claim breaks, so the README says what was and was not checked.
