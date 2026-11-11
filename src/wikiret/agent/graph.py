"""LangGraph pipeline: retrieve -> draft -> verify -> (answer | reformulate and retry once | refuse).

The model can say anything; what leaves the graph is decided by `verify` and `finish`.
The final answer is rebuilt from claims that passed verification, so unverified text
(including text a passage tried to get the model to emit) is not returned."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .generators import Generator, leaks_system_prompt
from .retrievers import Retriever
from .text import is_instruction_like
from .types import REFUSAL, AgentResult, Passage, Verdict
from .verify import VerifyConfig, verify


class State(TypedDict, total=False):
    question: str
    query: str
    attempt: int
    passages: list[Passage]
    draft: str
    verdict: Verdict
    result: AgentResult
    trace: list[dict]


def build_agent(retriever: Retriever, generator: Generator, *, k: int = 5, max_retries: int = 1,
                cfg: VerifyConfig = VerifyConfig()):
    def retrieve(s: State) -> State:
        passages = retriever.retrieve(s["query"], k)
        trace = s["trace"] + [{"step": "retrieve", "query": s["query"], "ids": [p.id for p in passages],
                               "quarantined_passages": [p.id for p in passages if is_instruction_like(p.text)]}]
        return {"passages": passages, "trace": trace}

    def draft(s: State) -> State:
        text = generator.draft(s["question"], s["passages"])
        return {"draft": text, "trace": s["trace"] + [{"step": "draft", "chars": len(text)}]}

    def check(s: State) -> State:
        v = verify(s["question"], s["draft"], s["passages"], cfg)
        if v.ok and leaks_system_prompt(s["draft"]):
            v = Verdict(False, v.checks)
        reasons = [c.reason for c in v.checks if not c.ok]
        return {"verdict": v, "trace": s["trace"] + [{"step": "verify", "ok": v.ok, "rejected": reasons}]}

    def route(s: State) -> str:
        if s["verdict"].ok:
            return "finish"
        return "reformulate" if s["attempt"] <= max_retries else "refuse"

    def reformulate(s: State) -> State:
        new = generator.reformulate(s["question"], s["attempt"])
        # asking the retriever the same thing twice cannot change its answer
        if new.strip().lower() == s["query"].strip().lower():
            return {"attempt": max_retries + 1, "trace": s["trace"] + [{"step": "reformulate", "query": new, "skipped": True}]}
        return {"query": new, "attempt": s["attempt"] + 1, "trace": s["trace"] + [{"step": "reformulate", "query": new}]}

    def after_reformulate(s: State) -> str:
        return "refuse" if s["trace"][-1].get("skipped") else "retrieve"

    def finish(s: State) -> State:
        ok_claims = [c for c in s["verdict"].checks if c.ok]
        answer = " ".join(f"{c.claim.rstrip('.!? ')} " + "".join(f"[{n}]" for n in c.citations) + "." for c in ok_claims)
        cited = sorted({n for c in ok_claims for n in c.citations})
        res = AgentResult(s["question"], answer, False, [s["passages"][n - 1].id for n in cited], s["attempt"], s["trace"])
        return {"result": res}

    def refuse(s: State) -> State:
        return {"result": AgentResult(s["question"], REFUSAL, True, [], s["attempt"], s["trace"])}

    g = StateGraph(State)
    for name, fn in (("retrieve", retrieve), ("draft", draft), ("verify", check), ("reformulate", reformulate),
                     ("finish", finish), ("refuse", refuse)):
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route, {"finish": "finish", "reformulate": "reformulate", "refuse": "refuse"})
    g.add_conditional_edges("reformulate", after_reformulate, {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("finish", END)
    g.add_edge("refuse", END)
    app = g.compile()

    def ask(question: str) -> AgentResult:
        out = app.invoke({"question": question, "query": question, "attempt": 1, "trace": []})
        return out["result"]

    return ask
