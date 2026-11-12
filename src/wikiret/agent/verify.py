"""Deterministic support check for a drafted answer.

Each claim sentence must (1) cite passages that exist, (2) be covered by a window of one
or two consecutive evidence sentences of a cited passage, (3) only use numbers and quoted
spans that appear in that passage, and (4) come from a passage that also covers the
question's own terms. Check 4 is what stops an answer from being 'supported' by a passage
that is on a neighbouring topic.

The check is lexical. It does not detect negation, entity swaps with overlapping words, or
paraphrase that shares few words with the source. See docs/adr/006."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .text import (
    clean_sentences,
    content_tokens,
    coverage,
    expects_number,
    has_number_or_date,
    is_instruction_like,
    numbers,
    question_tokens,
    quotes,
    strip_markers,
)
from .types import Claim, ClaimCheck, Passage, Verdict

_CITE = re.compile(r"\[\s*(\d+)\s*\]")
_CITE_AFTER_PUNCT = re.compile(r"([.!?])\s*((?:\[\s*\d+\s*\]\s*)+)")
_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class VerifyConfig:
    min_support: float = 0.7
    min_relevance: float = 0.5
    min_claim_tokens: int = 2
    # Optional learned answerability score for (question, passage text), for example the ONNX
    # cross-encoder. The lexical relevance check cannot tell a passage that answers from one that
    # only shares words (docs/adr/006); the threshold has to be calibrated on human labels, so
    # the default is off.
    answerability: Callable[[str, str], float] | None = None
    min_answerability: float = 0.0


def split_claims(draft: str) -> list[Claim]:
    # models tend to put the marker after the full stop; move it back so it stays with its sentence
    normalized = _CITE_AFTER_PUNCT.sub(lambda m: f" {m.group(2).strip()}{m.group(1)}", draft)
    claims = []
    for part in _SPLIT.split(normalized.strip()):
        if not part.strip():
            continue
        cites = tuple(sorted({int(c) for c in _CITE.findall(part)}))
        text = strip_markers(part)
        if text:
            claims.append(Claim(text, cites))
    return claims


def _windows(passage: Passage) -> list[str]:
    s = clean_sentences(passage.text)
    return s + [f"{a} {b}" for a, b in zip(s, s[1:])]


def score_claim(question: str, claim_text: str, passage: Passage) -> tuple[float, float]:
    """(support, relevance) of one claim against one passage."""
    ctoks = content_tokens(claim_text)
    support = max((coverage(ctoks, content_tokens(w)) for w in _windows(passage)), default=0.0)
    body = " ".join(clean_sentences(passage.text))
    relevance = coverage(question_tokens(question), content_tokens(f"{passage.title} {body}"))
    return support, relevance


def check_claim(question: str, c: Claim, passages: list[Passage], cfg: VerifyConfig = VerifyConfig()) -> ClaimCheck:
    if is_instruction_like(c.text):
        return ClaimCheck(c.text, c.citations, False, "claim looks like an instruction")
    if len(content_tokens(c.text)) < cfg.min_claim_tokens:
        return ClaimCheck(c.text, c.citations, False, "claim too short to verify")
    if not c.citations:
        return ClaimCheck(c.text, c.citations, False, "no citation")
    if any(not 1 <= n <= len(passages) for n in c.citations):
        return ClaimCheck(c.text, c.citations, False, "unknown citation")
    best = None
    for n in c.citations:
        p = passages[n - 1]
        support, rel = score_claim(question, c.text, p)
        body = " ".join(clean_sentences(p.text)).lower()
        facts_ok = numbers(c.text) <= numbers(body) and all(q.lower() in body for q in quotes(c.text))
        cand = (support, rel, facts_ok)
        if best is None or cand > best:
            best = cand
    support, rel, facts_ok = best
    if support < cfg.min_support:
        reason = "not supported by the cited passage"
    elif not facts_ok:
        reason = "number or quotation not found in the cited passage"
    elif expects_number(question) and not has_number_or_date(c.text):
        # 'when' and 'how many' questions need a figure or date; a sentence about the right
        # topic without one answers a different question
        reason = "answer type mismatch: a number or date is expected"
    elif rel < cfg.min_relevance:
        reason = "cited passage does not address the question"
    elif cfg.answerability is not None and not any(
            cfg.answerability(question, f"{passages[n - 1].title} {passages[n - 1].text}") >= cfg.min_answerability
            for n in c.citations):
        reason = "answerability score below threshold"
    else:
        reason = "supported"
    return ClaimCheck(c.text, c.citations, reason == "supported", reason, round(support, 3), round(rel, 3))


def judge_label(check: ClaimCheck, cfg: VerifyConfig = VerifyConfig()) -> str:
    """Three-way label comparable with the human rubric (docs/label_rubric.md), derived from the
    same checks the agent gates on, so validating the judge validates the gate."""
    if check.ok:
        return "supported"
    if check.support >= cfg.min_support or (check.support >= 0.4 and check.relevance >= cfg.min_relevance):
        return "partial"
    return "not_supported"


def verify(question: str, draft: str, passages: list[Passage], cfg: VerifyConfig = VerifyConfig()) -> Verdict:
    claims = split_claims(draft)
    if not claims:
        return Verdict(False, (ClaimCheck("", (), False, "empty draft"),))
    checks = tuple(check_claim(question, c, passages, cfg) for c in claims)
    return Verdict(all(k.ok for k in checks), checks)
