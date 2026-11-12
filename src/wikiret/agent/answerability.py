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


def default_verify_config(
    reranker_dir: str | Path = "artifacts/reranker_int8",
    config_path: str | Path = "configs/answerability.json",
    max_len: int = 256,
) -> tuple[VerifyConfig, dict]:
    """The verifier configuration the agent entry points use. The answerability gate is on whenever the
    threshold file and the INT8 cross-encoder are both present, and the returned dict says which it is so
    result files can record it. Without either file the lexical verifier runs alone."""
    rd, cp = Path(reranker_dir), Path(config_path)
    if not (cp.exists() and (rd / "model.onnx").exists()):
        return VerifyConfig(), {"answerability_gate": False}
    from ..rerank import OnnxReranker

    cfg = gate_config(OnnxReranker(rd, max_len=max_len), cp)
    return cfg, {"answerability_gate": True, "min_answerability": cfg.min_answerability}
