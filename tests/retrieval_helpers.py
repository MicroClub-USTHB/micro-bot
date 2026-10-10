"""Shared helpers for retrieval tests: a keyword-overlap reranker and a retriever factory."""

import re
import threading
import time

from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.settings import RetrievalSettings

_WORD = re.compile(r"\w+")


class FakeReranker:
    """Score = share of the query's words found in the passage (0..1); records concurrency."""

    fingerprint = "fake-reranker"

    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def score(self, query, passages):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.error:
                raise self.error
            time.sleep(self.delay)
            self.calls.append((query, list(passages)))
            words = set(_WORD.findall(query.lower()))
            return [len(words & set(_WORD.findall(p.lower()))) / len(words) for p in passages]
        finally:
            with self._lock:
                self.active -= 1


def make_retriever(pipe, reranker=None, **settings) -> HybridRetriever:
    s = RetrievalSettings(qdrant_collection=pipe.settings.qdrant_collection, **settings)
    return HybridRetriever.from_pipeline(pipe, s, reranker or FakeReranker())
