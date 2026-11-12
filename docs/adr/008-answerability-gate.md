# 008. Cross-encoder answerability gate for the verifier

Status: accepted, with modest validated accuracy

## Context
ADR 006 describes a lexical verifier. Scored against a labelled sheet it did not work on system answers: on 200 system-answer rows its agreement with the labels was at chance (kappa -0.006, interval -0.032 to 0.018). It called 194 of 200 answers supported, while the labels mark 48 supported, 76 partial and 76 not supported.

## Labels
300 rows (200 system answers, 100 control rows pairing a question with a passage from an unrelated question), labelled by me against docs/label_rubric.md as supported, partial or not supported, every row in two passes; the two passes agree on 84% of rows (kappa 0.73). One person labelled and that person is also the builder, so the gate is validated against a single reading.

The final labels are in `results/labels/answerability_labels.csv`: row id, stratum, question id, passage id and the label, with no passage text and no notes. `scripts/answerability_gate.py results/labels/answerability_labels.csv data/nq data/dev` looks the question and passage up by id and regenerates `results/answerability_gate.json` and `configs/answerability.json`; I reran it from the shipped labels and both files came out identical. The 84% and kappa 0.73 between the two passes cannot be recomputed from the repo, because the first pass is not shipped. `scripts/judge_cross_encoder.py` and `scripts/judge_agreement.py` still read the full sheet, which holds passage text and stays outside the repository.

## What I tried
The ONNX cross-encoder scores (question, passage) and (question, answer sentence). Cutoffs were fitted on half of the system rows and applied to the other half, then swapped (results/judge_cross_encoder.json):

| Score | AUC supported vs rest | AUC not supported vs rest | 3-class kappa (out of sample) |
|---|---|---|---|
| lexical judge | n/a | n/a | -0.006 |
| question + answer sentence | 0.678 | 0.743 | 0.245 [0.143, 0.348] |
| question + passage | 0.762 | 0.873 | 0.355 [0.251, 0.458] |

## Decision
Use the question and passage score as a refusal gate and leave the three-way judgement alone. Threshold from scripts/answerability_gate.py: refuse when the score is below 4.786. Over 40 random half splits (results/answerability_gate.json) the gate refused 75.3% of answers labelled not supported (sd 0.095) and wrongly refused 20.8% of answers labelled supported or partial (sd 0.073). The threshold in configs/answerability.json was chosen on all 200 rows, so only the cross-fitted rates are out-of-sample.

## What the gate does and does not check
It scores whether the cited passage is relevant to the question, and refuses when that score is low. It does not check that the answer sentence is true or follows from the passage, so a confident wrong sentence drawn from a relevant passage passes.

About a quarter of unsupported answers get through and about a fifth of acceptable ones are refused. The rates come from 200 system rows from the extractive generator on the reduced dev corpus, and they will move with a stronger generator. The partial label is where my two passes disagree most, and the gate cannot separate partial from supported.

## Inside the agent
The agent entry points turn the gate on when its files exist (ADR 006). Over the 300 smoke questions the gated agent answers 58 and refuses 242 (results/agent_smoke_gated.json), against 251 answered by the lexical verifier alone. The per-answer rates above do not predict that, because the agent applies the threshold to every claim of an answer and requires all of them to pass. The threshold was fitted on 200 answers from the same sampled questions, so the smoke run is in-sample. A threshold fitted per claim, or a rule that needs only one claim to clear it, would need its own labels.
