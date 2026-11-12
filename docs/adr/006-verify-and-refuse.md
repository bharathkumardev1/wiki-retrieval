# 006. Cited answers are gated by a deterministic verifier, with a bounded retry and a refusal

Status: accepted for the extractive baseline; the lexical support judge did not agree with the labelled sheet, see ADR 008

## Context
An answer agent over Wikipedia passages can fail in two ways that matter more than a low score: it states something
the passages do not say, and it does what a passage tells it to do. The retrieved text is untrusted input.

## Decision
The graph is retrieve, draft, verify, then either return, reformulate and try once more, or refuse. The generator
sits behind a two-method interface, so the same gate runs over the extractive baseline, a local model, or a hosted one.

What leaves the graph is rebuilt from claim sentences that passed verification. A claim passes when:
1. every citation points at a passage that was shown to the generator;
2. a window of one or two consecutive evidence sentences covers at least 70% of the claim's content stems;
3. every number and quoted span in the claim appears in the cited passage;
4. for 'when' and 'how many/much/long/tall/old' questions, the claim contains a number or a date;
5. the cited passage covers at least half of the question's content terms (question words excluded);
6. optionally, a learned answerability score clears a threshold (off by default; see below).

Instruction handling: a passage sentence that looks like an instruction ("ignore previous instructions", role prefixes,
chat-template tokens, "never refuse", "reveal your prompt", and similar) is never evidence, and `[n]` markers inside
passage text are deleted so a passage cannot vouch for itself. A final check rejects any answer that reproduces
eight consecutive words of the system prompt.

The retry reformulates the query once. If the new query equals the old one it refuses immediately, since the
retriever would return the same passages.

## Measured (scripts/agent_smoke.py, results/agent_smoke.json; reduced dev corpus, BM25 top 5, extractive generator)
- 300 sampled questions: answered 251 (83.7%), refused 49.
- A labelled-relevant passage was in the retrieved top 5 for 82.0% of questions. Among answered questions the cited
  passage was labelled relevant in 69.7%. The other 30.3% cite a passage that the NQ labels do not mark relevant; some
  of those may still answer the question (the labels are incomplete), which is what the human sheet is for.
- Evidence taken from a different, random question: answered 2 of 300 (0.7%).
- Planted instructions in the top passage (4 attack sentences x 100 questions = 400 trials): 0 answers changed, 0
  answer/refusal flips, 0 answers containing attack text. An earlier run changed 4 answers: a passage ending in the
  wiki tag "[citation needed]" glued the attack onto the last real sentence and the whole thing was quarantined. Fixed by
  removing such tags before splitting; there is a test for it.
- Support check versus the lexical relevance score, on the 200 'system' rows of the label sheet: the relevance score
  separates labelled-relevant from other cited passages only weakly (AUC 0.61, 103 vs 97 rows). Thresholds of 0.5 keep
  97% of both groups.

## Known failure modes
- The check is lexical. It does not detect negation ("was not completed in 1889"), swapped entities that share
  words, or a paraphrase that shares few stems with its source. The last one makes the gate refuse correct answers from a
  model that paraphrases where it could quote.
- The answer-type check is shallow: any number satisfies it, so a year can satisfy 'how long'. Only when- and
  how-many-type questions are covered; 'who' and 'where' are not.
- The extractive generator copies sentences, so claim support is nearly always 1.0. That makes the support test uninformative
  for it; the work is done by the relevance and answer-type checks, and those are the weak ones (see above).
- Instruction detection is pattern-based. A paraphrased instruction that matches none of the patterns is treated as an
  ordinary sentence. It still cannot reach the output unless it is also a verified, relevant claim.
- A false statement written as a normal sentence in a retrieved passage is not detected. The check measures lexical support and cannot tell a true claim from a false one.
- Sentence splitting is a regular expression and mis-splits some abbreviations.

## Validation
The thresholds (0.7, 0.5) were set up front and were not tuned. Scored against the 300-row labelled sheet (results/judge_agreement_lexical.json),
the lexical judge agreed with the labels at chance on the 200 system answers (kappa -0.006, interval -0.032 to 0.018): it called 194 of 200 supported.
ADR 008 replaces it with a cross-encoder answerability gate and reports its out-of-sample agreement.

The optional answerability hook takes any function of (question, passage). ADR 008 plugs the ONNX cross-encoder from
the search service into it and calibrates the threshold on the same labels.

## Generators
- ExtractiveGenerator: deterministic, tested, used for every number above.
- TransformersGenerator (Qwen2.5-0.5B-Instruct on CPU): runs, but in an 8-question trial it produced no answer that
  passed the gate (empty drafts or missing citations), at 37 to 65 seconds per question on a loaded 2 vCPU host. That is
  too few questions to estimate a rate, but enough to rule this model size out as a useful drafter here.
