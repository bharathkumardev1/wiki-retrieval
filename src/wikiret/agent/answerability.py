"""Answerability gate for the verifier: a cross-encoder score of (question, passage)."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .verify import VerifyConfig


def gate_config(
    score: Callable[[str, list[str]], object],
    config_path: str | Path = "configs/answerability.json",
    base: VerifyConfig | None = None,
) -> VerifyConfig:
    """VerifyConfig whose answerability check uses `score(query, [passage])`, with the threshold
    picked from the labelled sheet (scripts/answerability_gate.py, docs/adr/008)."""
    threshold = json.loads(Path(config_path).read_text())["min_answerability"]
    b = base or VerifyConfig()
    return VerifyConfig(min_support=b.min_support, min_relevance=b.min_relevance, min_claim_tokens=b.min_claim_tokens,
                        answerability=lambda q, text: float(score(q, [text])[0]), min_answerability=threshold)
