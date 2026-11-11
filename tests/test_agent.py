import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from wikiret.agent import (REFUSAL, ExtractiveGenerator, FromLangChain, Passage, ServiceRetriever, ToLangChain,
                           VerifyConfig, build_agent, judge_label, verify)
from wikiret.agent.verify import split_claims

EIFFEL = Passage("p1", "Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889. "
                                        "It is 330 metres tall.")
BERLIN = Passage("p2", "Berlin Wall", "The Berlin Wall fell on 9 November 1989. It had divided the city for 28 years.")
NOISE = Passage("p3", "Bananas", "Bananas are a good source of potassium. They are botanically berries.")


class FakeRetriever:
    def __init__(self, table, default=()):
        self.table, self.default, self.calls = table, list(default), []

    def retrieve(self, query, k):
        self.calls.append(query)
        return list(self.table.get(query, self.default))[:k]


class ScriptedGenerator:
    """Returns canned drafts in order, to drive the graph without a model."""

    def __init__(self, drafts, rewrite="eiffel tower completed"):
        self.drafts, self.rewrite, self.n = list(drafts), rewrite, 0

    def draft(self, question, passages):
        self.n += 1
        return self.drafts[min(self.n, len(self.drafts)) - 1]

    def reformulate(self, question, attempt):
        return self.rewrite


def test_extractive_answer_is_cited_and_verified():
    ask = build_agent(FakeRetriever({}, [EIFFEL, BERLIN, NOISE]), ExtractiveGenerator())
    r = ask("When was the Eiffel Tower completed?")
    assert not r.refused and "1889" in r.answer and r.cited_passage_ids == ["p1"]
    assert r.answer.endswith("[1].")


def test_refuses_when_passages_do_not_contain_the_answer():
    ask = build_agent(FakeRetriever({}, [BERLIN, NOISE]), ExtractiveGenerator())
    r = ask("When was the Eiffel Tower completed?")
    assert r.refused and r.answer == REFUSAL and r.cited_passage_ids == []


def test_retry_with_reformulated_query_recovers_once():
    first, second = "when was that tall iron tower completed", "eiffel tower completed"
    ret = FakeRetriever({first: [NOISE], second: [EIFFEL, NOISE]})
    gen = ScriptedGenerator(["Bananas were first grown in 1500 [1].", "The Eiffel Tower was completed in 1889 [1]."], rewrite=second)
    r = build_agent(ret, gen)(first)
    assert ret.calls == [first, second]
    assert not r.refused and r.attempts == 2 and r.cited_passage_ids == ["p1"]


def test_retry_is_bounded():
    ret = FakeRetriever({}, [NOISE])
    gen = ScriptedGenerator(["The tower is 330 metres tall [1]."], rewrite="something else")
    r = build_agent(ret, gen, max_retries=1)("how tall is the tower")
    assert r.refused and len(ret.calls) == 2


def test_no_retry_when_reformulation_is_identical():
    ret = FakeRetriever({}, [NOISE])
    gen = ScriptedGenerator(["The tower is 330 metres tall [1]."], rewrite="How tall is the tower")
    r = build_agent(ret, gen)("how tall is the tower")
    assert r.refused and len(ret.calls) == 1


WHEN = "When was the Eiffel Tower completed?"


@pytest.mark.parametrize("question,draft,reason", [
    (WHEN, "The Eiffel Tower was completed in 1889.", "no citation"),
    (WHEN, "The Eiffel Tower was completed in 1889 [7].", "unknown citation"),
    (WHEN, "The Eiffel Tower was completed in 1890 [1].", "number or quotation"),
    (WHEN, "The Eiffel Tower is made entirely of chocolate and sits in Rome in 1889 [1].", "not supported"),
    ("What is the Eiffel Tower made of?", "Bananas are a good source of potassium [2].", "does not address the question"),
])
def test_verifier_rejects(question, draft, reason):
    v = verify(question, draft, [EIFFEL, NOISE])
    assert not v.ok and reason in v.checks[0].reason


def test_verifier_accepts_marker_after_full_stop_and_two_sentence_window():
    v = verify("When was the Eiffel Tower completed?", "The Eiffel Tower in Paris was completed in 1889. [1]", [EIFFEL])
    assert v.ok
    claims = split_claims("First fact [1]. Second fact [2][1].")
    assert [c.citations for c in claims] == [(1,), (1, 2)]


def test_quoted_span_must_appear_verbatim():
    p = Passage("q", "Robert Frost", 'Frost published "The Road Not Taken" in 1916. It is widely read.')
    q = "When did Frost publish The Road Not Taken?"
    ok = verify(q, 'Frost published "The Road Not Taken" in 1916 [1].', [p])
    # enough shared words to pass the overlap test, so only the quotation check can catch this
    bad = verify(q, 'Frost published "The Road Not Taken Again" in 1916 [1].', [p])
    assert ok.ok and not bad.ok and "quotation" in bad.checks[0].reason


def test_number_question_rejects_an_on_topic_sentence_without_a_number():
    on_topic = verify("When was the Eiffel Tower completed?", "The Eiffel Tower is a lattice tower in Paris [1].", [EIFFEL])
    assert not on_topic.ok and "answer type" in on_topic.checks[0].reason


def test_judge_label_follows_the_gate():
    from wikiret.agent import check_claim
    from wikiret.agent.verify import Claim
    one = [EIFFEL]
    q = "When was the Eiffel Tower completed?"
    assert judge_label(check_claim(q, Claim("It was completed in 1889", (1,)), one)) == "supported"
    # on topic and quoted from the passage, but it does not give the date that was asked for
    assert judge_label(check_claim(q, Claim("The Eiffel Tower is a wrought-iron lattice tower in Paris", (1,)), one)) == "partial"
    assert judge_label(check_claim(q, Claim("Bananas are botanically berries in 1500", (1,)), one)) == "not_supported"


class StubRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return [Document(page_content=EIFFEL.text, metadata={"id": "p1", "title": EIFFEL.title}),
                Document(page_content=NOISE.text, metadata={"id": "p3", "title": NOISE.title})]


def test_langchain_adapters_round_trip():
    ask = build_agent(FromLangChain(StubRetriever()), ExtractiveGenerator())
    assert ask("When was the Eiffel Tower completed?").cited_passage_ids == ["p1"]
    lc = ToLangChain(inner=FakeRetriever({}, [EIFFEL, BERLIN]), k=1)
    docs = lc.invoke("anything")
    assert len(docs) == 1 and docs[0].metadata["id"] == "p1"


def test_service_retriever_reads_search_response():
    app = FastAPI()

    @app.post("/v1/search")
    def search(body: dict):
        assert body["k"] == 2
        return {"results": [{"id": "p1", "title": "Eiffel Tower", "text": EIFFEL.text}, {"id": "p3", "text": NOISE.text}]}

    got = ServiceRetriever(TestClient(app)).retrieve("q", 2)
    assert [p.id for p in got] == ["p1", "p3"] and got[1].title == ""


def test_modal_may_is_not_a_date():
    from wikiret.agent.text import has_number_or_date
    assert not has_number_or_date("The phrase may refer to a hoop")
    assert has_number_or_date("It opened in May and closed in December") and has_number_or_date("On 4 May the wall fell")


def test_optional_answerability_scorer_can_veto_a_lexically_fine_claim():
    draft = "The Eiffel Tower was completed in 1889 [1]."
    assert verify(WHEN, draft, [EIFFEL]).ok
    strict = VerifyConfig(answerability=lambda q, p: -1.0, min_answerability=0.0)
    v = verify(WHEN, draft, [EIFFEL], strict)
    assert not v.ok and "answerability" in v.checks[0].reason
    lenient = VerifyConfig(answerability=lambda q, p: 3.0, min_answerability=0.0)
    assert verify(WHEN, draft, [EIFFEL], lenient).ok
