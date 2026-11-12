# 008. Cross-encoder answerability gate for the verifier

Status: accepted, with modest validated accuracy

## Context
ADR 006 describes a lexical verifier. Scored against a labelled sheet it did not work on system answers: on 200 system-answer rows its agreement with the labels was at chance (kappa -0.006, interval -0.032 to 0.018). It called 194 of 200 answers supported, while the labels mark 48 supported, 76 partial and 76 not supported.

## Labels
300 rows (200 system answers, 100 control rows pairing a question with a passage from an unrelated question), labelled by the project owner against docs/label_rubric.md as supported, partial or not supported. The owner labelled every row in two passes; the two passes agree on 84% of rows (kappa 0.73). There is one annotator, and that person is also the builder, so this validates the gate against one person's reading, not against independent raters. The sheet contains passage text and is kept outside the repository.

## What I tried
The ONNX cross-encoder scores (question, passage) and (question, answer sentence). Cutoffs were fitted on half of the system rows and applied to the other half, then swapped (results/judge_cross_encoder.json):

| Score | AUC supported vs rest | AUC not supported vs rest | 3-class kappa (out of sample) |
|---|---|---|---|
| lexical judge | n/a | n/a | -0.006 |
| question + answer sentence | 0.678 | 0.743 | 0.245 [0.143, 0.348] |
| question + passage | 0.762 | 0.873 | 0.355 [0.251, 0.458] |

## Decision
Use the question and passage score as a refusal gate, not as a three-way judge. Threshold from scripts/answerability_gate.py: refuse when the score is below 4.786. Over 40 random half splits (results/answerability_gate.json) the gate refused 75.3% of answers labelled not supported (sd 0.095) and wrongly refused 20.8% of answers labelled supported or partial (sd 0.073). The threshold in configs/answerability.json was chosen on all 200 rows, so only the cross-fitted rates are out-of-sample.

## Limits
- It still lets about a quarter of unsupported answers through and refuses about a fifth of acceptable ones.
- It judges whether the passage answers the question, not whether the answer sentence is true or follows from it.
- One annotator, 200 system rows, answers from the extractive generator on the reduced dev corpus. The rates will move with a stronger generator.
- The rubric's partial label is where the annotator disagrees with themselves most, and the gate cannot separate partial from supported.
