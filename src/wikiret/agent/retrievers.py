"""Retriever interface and adapters. LangChain is used only at this boundary, so any
LangChain retriever can feed the agent and the agent's retriever can be used elsewhere."""
from __future__ import annotations

from typing import Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .types import Passage


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Passage]: ...


class ServiceRetriever:
    """Calls the search service. The service returns a 400-character snippet per passage, so
    that snippet is the evidence the verifier sees."""

    def __init__(self, client, path: str = "/v1/search") -> None:
        self.client, self.path = client, path

    def retrieve(self, query: str, k: int) -> list[Passage]:
        r = self.client.post(self.path, json={"query": query, "k": k})
        r.raise_for_status()
        return [Passage(x["id"], x.get("title", ""), x["text"]) for x in r.json()["results"]]


class FromLangChain:
    """Use any LangChain BaseRetriever as an agent retriever. Document metadata `id` and
    `title` are read when present."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, k: int) -> list[Passage]:
        docs = self.retriever.invoke(query)[:k]
        return [Passage(str(d.metadata.get("id", i)), str(d.metadata.get("title", "")), d.page_content)
                for i, d in enumerate(docs)]


class ToLangChain(BaseRetriever):
    inner: object
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=p.text, metadata={"id": p.id, "title": p.title})
                for p in self.inner.retrieve(query, self.k)]
