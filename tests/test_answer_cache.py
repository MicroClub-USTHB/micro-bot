"""A repeated first question on the same corpus is answered from the cache: 0 LLM calls."""

from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from micro_bot.answer_cache import AnswerCache, cache_key
from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage
from micro_bot.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeHealthProbe,
    FakeLanguageDetector,
    create_app,
)
from micro_bot.schemas import Chunk, Retrieval, Source

KEY = {"Authorization": "Bearer api-secret"}


@dataclass
class Generator:
    text: str = "Friday. [S1]"
    calls: int = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        return GenerationResult(self.text, GenerationUsage(1, 5, 2))


@dataclass
class Retrieve:
    result: Retrieval = field(
        default_factory=lambda: Retrieval(
            chunks=[Chunk(chunk_id="c", doc_id="d", title="Guide", text="Friday.", score=0.9)],
            top_score=0.9,
            decision="answer",
        )
    )

    async def retrieve(self, query, history, *, rewrite=None) -> Retrieval:
        return self.result


def build(tmp_path: Path, generator: Generator, documents=None) -> tuple[TestClient, object]:
    settings = Settings(data_dir=tmp_path, api_key="api-secret")
    state = ApplicationState(
        settings=settings,
        documents=documents or FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=Retrieve(),
        generator=generator,
        language_detector=FakeLanguageDetector(),
    )
    app = create_app(settings=settings, state=state)
    return TestClient(app, raise_server_exceptions=False), state


def ask(client: TestClient, message: str, **extra) -> dict:
    return client.post("/api/v1/chat", json={"message": message, **extra}, headers=KEY).json()


def test_same_question_twice_costs_one_call(tmp_path: Path) -> None:
    generator = Generator()
    client, _ = build(tmp_path, generator)

    first = ask(client, "When is it?")
    second = ask(client, "  when IS it? ")

    assert generator.calls == 1
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]
    assert second["grounded"] is True and second["usage"]["llm_calls"] == 0
    assert second["conversation_id"] != first["conversation_id"]


def test_a_document_change_bypasses_old_answers(tmp_path: Path) -> None:
    generator = Generator()
    documents = FakeDocumentBackend()
    client, _ = build(tmp_path, generator, documents)

    ask(client, "When is it?")
    documents._corpus_version += 1  # an upload, replace or delete
    ask(client, "When is it?")

    assert generator.calls == 2


def test_follow_ups_are_never_cached(tmp_path: Path) -> None:
    generator = Generator()
    client, _ = build(tmp_path, generator)

    cid = ask(client, "When is it?")["conversation_id"]
    ask(client, "When is it?", conversation_id=cid)

    assert generator.calls == 2


def test_refusals_are_not_cached(tmp_path: Path) -> None:
    generator = Generator(text="NOT_FOUND")
    client, _ = build(tmp_path, generator)

    ask(client, "Fee?")
    ask(client, "Fee?")

    assert generator.calls == 2


def test_streamed_question_uses_the_cache(tmp_path: Path) -> None:
    generator = Generator()
    client, _ = build(tmp_path, generator)

    ask(client, "When is it?")
    body = client.post(
        "/api/v1/chat", json={"message": "When is it?", "stream": True}, headers=KEY
    ).text

    assert generator.calls == 1
    assert "event: done" in body and '"llm_calls": 0' in body


def test_cache_evicts_the_oldest_entry() -> None:
    cache = AnswerCache(size=1)
    source = Source(doc_id="d", title="t", snippet="s")

    cache.put(cache_key("a", "en", "1"), "A", [source])
    cache.put(cache_key("b", "en", "1"), "B", [source])

    assert cache.get(cache_key("a", "en", "1")) is None
    assert cache.get(cache_key("b", "en", "1")) == ("B", [source])


def test_size_zero_disables_the_cache() -> None:
    cache = AnswerCache(size=0)
    cache.put(cache_key("a", "en", "1"), "A", [])

    assert cache.get(cache_key("a", "en", "1")) is None


def test_a_citation_only_reply_is_a_fallback_and_never_cached(tmp_path: Path) -> None:
    generator = Generator(text="[S1]")
    client, _ = build(tmp_path, generator)

    first = ask(client, "What is it?")
    ask(client, "What is it?")

    assert first["grounded"] is False and first["answer"].strip() != "[Guide]"
    assert generator.calls == 2
