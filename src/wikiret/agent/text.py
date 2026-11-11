"""Text helpers shared by the generator and the verifier.

Retrieved passages are untrusted input. Two defences live here: instruction-shaped
sentences are never eligible as evidence, and citation-looking markers inside passage
text are removed so a passage cannot vouch for itself."""
from __future__ import annotations

import re

from ..bm25 import analyze

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_MARKER = re.compile(r"\[\s*\d+\s*\]")
# editorial tags left in Wikipedia text, e.g. "[citation needed]"; they glue sentences together and are not content
_WIKI_TAG = re.compile(r"\[(?:citation|clarification|dubious|verification|when|who|where|according to whom)[^\]]*\]", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTED = re.compile(r"\"([^\"]{8,})\"")

# Deliberately broad: a false positive only removes one sentence from the evidence pool,
# while a false negative lets a passage steer the answer.
_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|your)\b.{0,30}\b(instruction|prompt|rule|direction)"
    r"|\b(system|assistant|developer)\s*(prompt|message|instruction)"
    r"|^\s*(system|assistant|user)\s*:"
    r"|<\|?\s*(im_start|im_end|system|assistant|endoftext)"
    r"|\b(do not|don't|never)\s+(refuse|decline)\b"
    r"|\balways\s+(answer|respond|reply|cite)\b"
    r"|\b(reveal|print|repeat|output|show|leak)\b.{0,30}\b(prompt|instructions)\b"
    r"|\byou\s+(must|should|are now|will)\b"
    r"|\b(new|updated)\s+instructions?\b",
    re.IGNORECASE | re.DOTALL,
)


def is_instruction_like(sentence: str) -> bool:
    return bool(_INJECTION.search(sentence))


def strip_markers(text: str) -> str:
    out = re.sub(r"\s+", " ", _WIKI_TAG.sub("", _MARKER.sub("", text))).strip()
    return re.sub(r"\s+([.!?,;])", r"\1", out)  # removing "[1]" before a full stop leaves a gap


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(strip_markers(text)) if s.strip()]


def clean_sentences(text: str) -> list[str]:
    """Sentences of a passage that are allowed to serve as evidence."""
    return [s for s in sentences(text) if not is_instruction_like(s)]


def content_tokens(text: str) -> set[str]:
    return set(analyze(strip_markers(text)))


def coverage(needles: set[str], haystack: set[str]) -> float:
    return len(needles & haystack) / len(needles) if needles else 0.0


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def quotes(text: str) -> list[str]:
    return _QUOTED.findall(text)


_WH = frozenset({"who", "whom", "whose", "when", "where", "what", "which", "why", "how", "many", "much", "did", "does", "do"})
# Case-sensitive on purpose: lowercase "may" is a verb. "May" only counts next to a day, a year, or "in/of/since".
_MONTH = re.compile(r"\b(?:January|February|March|April|June|July|August|September|October|November|December)\b"
                    r"|\b(?:in|of|on|since|until|from|early|late|mid)\s+May\b|\bMay\s+\d|\d\s+May\b")
_NUMERIC_Q = re.compile(r"^\W*(?:in\s+what\s+year|what\s+year|when|how\s+(?:many|much|long|tall|old|far|big|high|large|deep|wide|heavy))\b", re.I)


def question_tokens(question: str) -> set[str]:
    """Content terms of a question. Question words are dropped: no passage 'contains' the word
    'who', so counting it would penalise every correct passage."""
    return {t for t in content_tokens(question) if t not in _WH}


def expects_number(question: str) -> bool:
    return bool(_NUMERIC_Q.search(question))


def has_number_or_date(text: str) -> bool:
    return bool(_NUMBER.search(text) or _MONTH.search(text))
