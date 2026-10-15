"""Chat route with the query rewrite (query-rewrite Req 1.1, 1.4, 1.7)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from mcclub_rag.config import Settings
from mcclub_rag.llm.client import GenerationResult, GenerationUsage
from mcclub_rag.llm.quota import DailyQuota
from mcclub_rag.llm.rewrite import QueryRewriter
from mcclub_rag.main import ApplicationState, FakeDocumentBackend, FakeHealthProbe, create_app
from mcclub_rag.schemas import Chunk, Retrieval

REWRITE = {"lang": "arabizi", "query": "wach kayen parking?", "en": "Is there parking?"}
CHUNK = Chunk(chunk_id="c1", doc_id="d1", title="Guide", text="Parking: yes.", score=0.9)


@dataclass
class Retriever:
    rewrites: list = field(default_factory=list)

    async def retrieve(self, query, role, history, *, rewrite=None) -> Retrieval:
        self.rewrites.append(rewrite)
        return Retrieval(chunks=[CHUNK], top_score=0.9, decision="answer")


@dataclass
class Generator:
    """First call of a question is the rewrite (JSON), the second the answer."""

    rewrite_reply: GenerationResult
    prompts: list = field(default_factory=list)

    async def generate(self, system, user) -> GenerationResult:
        self.prompts.append(system)
        if "search queries" in system:
            return self.rewrite_reply
        return GenerationResult("Yes [S1].", GenerationUsage(1, 1000, 10))


def client(tmp_path: Path, reply: GenerationResult, quota: DailyQuota | None = None):
    retriever = Retriever()
    generator = Generator(reply)
    state = ApplicationState(
        settings=Settings(data_dir=tmp_path),
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=retriever,
        generator=generator,
        rewriter=QueryRewriter(timeout_s=5),
        quota=quota,
    )
    return TestClient(create_app(settings=state.settings, state=state)), retriever, generator


def test_rewrite_reaches_retrieval_and_usage_counts_both_calls(tmp_path):
    ok = GenerationResult(json.dumps(REWRITE), GenerationUsage(1, 300, 60))
    api, retriever, _ = client(tmp_path, ok)
    body = api.post("/api/v1/chat", json={"message": "wach kayen parking?"}).json()
    assert retriever.rewrites[0].translations == {"en": "Is there parking?"}
    assert body["usage"] == {"llm_calls": 2, "prompt_tokens": 1300, "completion_tokens": 70}
    assert body["grounded"] is True


def test_failed_rewrite_searches_as_typed(tmp_path):
    failed = GenerationResult(None, GenerationUsage(1), "timeout")
    api, retriever, _ = client(tmp_path, failed)
    body = api.post("/api/v1/chat", json={"message": "parking?"}).json()
    assert retriever.rewrites == [None]
    assert body["usage"]["llm_calls"] == 2  # the failed attempt is a provider call too
    assert body["grounded"] is True


def test_rewrite_skipped_when_only_one_quota_slot_is_left(tmp_path):
    quota = DailyQuota(tmp_path, limit=1, reserve=0)
    ok = GenerationResult(json.dumps(REWRITE), GenerationUsage(1, 300, 60))
    api, retriever, generator = client(tmp_path, ok, quota)
    body = api.post("/api/v1/chat", json={"message": "parking?"}).json()
    assert retriever.rewrites == [None]
    assert body["usage"]["llm_calls"] == 1  # the answer kept its slot
    assert len(generator.prompts) == 1


def test_blank_message_makes_no_rewrite_call(tmp_path):
    ok = GenerationResult(json.dumps(REWRITE), GenerationUsage(1))
    api, retriever, generator = client(tmp_path, ok)
    body = api.post("/api/v1/chat", json={"message": "   "}).json()
    assert body["usage"]["llm_calls"] == 0
    assert not generator.prompts and not retriever.rewrites
