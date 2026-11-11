from __future__ import annotations

from dataclasses import dataclass, field

REFUSAL = "The retrieved passages do not support an answer."


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[int, ...]  # 1-based positions in the passage list shown to the generator


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    citations: tuple[int, ...]
    ok: bool
    reason: str  # "supported" or why it was rejected
    support: float = 0.0
    relevance: float = 0.0


@dataclass(frozen=True)
class Verdict:
    ok: bool
    checks: tuple[ClaimCheck, ...]


@dataclass
class AgentResult:
    question: str
    answer: str  # REFUSAL when refused
    refused: bool
    cited_passage_ids: list[str] = field(default_factory=list)
    attempts: int = 1
    trace: list[dict] = field(default_factory=list)
