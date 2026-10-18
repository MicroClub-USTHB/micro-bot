"""``stream=true`` chat: SSE events, the live citation rewrite and the stream guards."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from micro_bot.answer import StreamFilter
from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage, OpenRouterClient
from micro_bot.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeHealthProbe,
    FakeLanguageDetector,
    create_app,
)
from micro_bot.messages import NO_ANSWER
from micro_bot.schemas import Chunk, Retrieval

KEY = {"Authorization": "Bearer api-secret"}
SYSTEM = "You are Micro Bot, the Micro Club assistant. Answer only from the retrieved passages."


def chunk(doc_id: str = "doc-1", title: str = "Club guide") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-c", doc_id=doc_id, title=title, text="Registration closes Friday.",
        score=0.9,
    )  # fmt: skip


@dataclass
class FakeRetrieval:
    result: Retrieval = field(
        default_factory=lambda: Retrieval(chunks=[chunk()], top_score=0.9, decision="answer")
    )

    async def retrieve(self, query, history, *, rewrite=None) -> Retrieval:
        return self.result


@dataclass
class StreamingGenerator:
    deltas: list[str]
    error: str | None = None
    streams: int = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        raise AssertionError("stream=true must not use generate")

    async def stream(self, system: str, user: str) -> AsyncIterator[str | GenerationResult]:
        self.streams += 1
        for delta in self.deltas:
            yield delta
        text = None if self.error else "".join(self.deltas)
        yield GenerationResult(text, GenerationUsage(1, 10, 5), self.error)


def build(tmp_path: Path, generator, retrieval=None) -> TestClient:
    settings = Settings(data_dir=tmp_path, api_key="api-secret")
    state = ApplicationState(
        settings=settings,
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=retrieval or FakeRetrieval(),
        generator=generator,
        language_detector=FakeLanguageDetector(),
    )
    return TestClient(create_app(settings=settings, state=state), raise_server_exceptions=False)


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_sends_meta_deltas_then_done_with_the_full_response(tmp_path: Path) -> None:
    generator = StreamingGenerator(["Registration ", "closes ", "Friday. [S", "1]"])
    response = build(tmp_path, generator).post(
        "/api/v1/chat", json={"message": "When?", "stream": True}, headers=KEY
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response)
    names = [name for name, _ in got]
    assert names[0] == "meta" and names[-1] == "done" and "delta" in names
    streamed = "".join(data["text"] for name, data in got if name == "delta")
    done = got[-1][1]
    assert streamed == "Registration closes Friday. [Club guide]"
    assert done["answer"] == streamed
    assert done["grounded"] is True
    assert done["sources"][0]["doc_id"] == "doc-1"
    assert done["usage"] == {"llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}
    assert done["conversation_id"] == got[0][1]["conversation_id"]


def test_stream_false_is_the_default_json_answer(tmp_path: Path) -> None:
    @dataclass
    class Plain:
        async def generate(self, system: str, user: str) -> GenerationResult:
            return GenerationResult("Friday. [S1]", GenerationUsage(1, 1, 1))

    response = build(tmp_path, Plain()).post("/api/v1/chat", json={"message": "When?"}, headers=KEY)

    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["answer"] == "Friday. [Club guide]"


def test_stream_needs_the_api_key_before_any_event(tmp_path: Path) -> None:
    response = build(tmp_path, StreamingGenerator(["x"])).post(
        "/api/v1/chat", json={"message": "When?", "stream": True}
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_streamed_refusal_never_shows_the_marker(tmp_path: Path) -> None:
    generator = StreamingGenerator(["NOT_", "FOUND [S1]"])
    got = events(
        build(tmp_path, generator).post(
            "/api/v1/chat", json={"message": "Fee?", "stream": True}, headers=KEY
        )
    )

    deltas = "".join(data["text"] for name, data in got if name == "delta")
    assert "NOT_FOUND" not in deltas
    assert deltas == NO_ANSWER["en"]
    assert got[-1][1]["grounded"] is False and got[-1][1]["sources"] == []


def test_provider_error_mid_stream_falls_back_in_done(tmp_path: Path) -> None:
    generator = StreamingGenerator(["Registration "], error="provider_error")
    got = events(
        build(tmp_path, generator).post(
            "/api/v1/chat", json={"message": "When?", "stream": True}, headers=KEY
        )
    )

    done = got[-1]
    assert done[0] == "done"
    assert done[1]["grounded"] is False  # the extractive fallback, authoritative over the deltas
    assert done[1]["usage"]["llm_calls"] == 1


def test_retrieval_failure_becomes_an_error_event(tmp_path: Path) -> None:
    class Down:
        async def retrieve(self, query, history, *, rewrite=None):
            raise type("IndexUnavailable", (Exception,), {"code": "index_unavailable"})("down")

    response = build(tmp_path, StreamingGenerator(["x"]), Down()).post(
        "/api/v1/chat", json={"message": "When?", "stream": True}, headers=KEY
    )

    got = events(response)
    assert response.status_code == 200
    assert got[-1] == ("error", {"error": {"code": "index_unavailable", "message": "down"}})


def test_generator_without_stream_still_streams_one_delta(tmp_path: Path) -> None:
    @dataclass
    class Plain:
        async def generate(self, system: str, user: str) -> GenerationResult:
            return GenerationResult("Friday. [S1]", GenerationUsage(1, 1, 1))

    got = events(
        build(tmp_path, Plain()).post(
            "/api/v1/chat", json={"message": "When?", "stream": True}, headers=KEY
        )
    )

    assert [name for name, _ in got] == ["meta", "delta", "done"]
    assert got[1][1]["text"] == got[2][1]["answer"] == "Friday. [Club guide]"


def feed_all(f: StreamFilter, deltas: list[str]) -> str:
    return "".join(f.feed(d) for d in deltas) + f.finish()


def test_filter_rewrites_split_citations_and_drops_unknown_ones() -> None:
    f = StreamFilter([chunk(), chunk("doc-2", "Bylaws [2025]")], SYSTEM)

    out = feed_all(f, ["See ", "【S", "2】 and [S", "1, S2] or [S9]."])

    assert out == "See [Bylaws (2025)] and [Club guide; Bylaws (2025)] or ."


def test_filter_holds_text_until_a_word_ends() -> None:
    f = StreamFilter([chunk()], SYSTEM)

    assert f.feed("Regis") == ""
    assert f.feed("tration closes") == "Registration "
    assert f.finish() == "closes"


def test_filter_keeps_an_unclosed_bracket_as_text() -> None:
    f = StreamFilter([chunk()], SYSTEM)

    assert feed_all(f, ["a [note ", "without end"]) == "a [note without end"


def test_filter_withholds_a_prompt_leak() -> None:
    f = StreamFilter([chunk()], SYSTEM)

    out = feed_all(f, ["You are Micro Bot, ", "the Micro Club assistant. ", "Answer only"])

    assert "Micro Club assistant" not in out


class _Chunk(SimpleNamespace):
    pass


class _Stream:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for c in self.chunks:
            yield c


def _client(stream: _Stream, calls: list):
    async def create(**kwargs):
        calls.append(kwargs)
        return stream

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _delta(text: str | None, finish: str | None = None) -> _Chunk:
    choice = SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)
    return _Chunk(choices=[choice], usage=None, model_extra={})


async def _collect(client: OpenRouterClient) -> list:
    return [item async for item in client.stream("sys", "user")]


async def test_client_streams_deltas_then_a_result_with_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    stream = _Stream([_delta("Hel"), _delta("lo"), _Chunk(choices=[], usage=usage, model_extra={})])
    calls: list = []
    client = OpenRouterClient(
        Settings(openrouter_api_key="k"), async_client_factory=lambda **_: _client(stream, calls)
    )

    items = await _collect(client)

    assert items[:2] == ["Hel", "lo"]
    assert items[-1] == GenerationResult("Hello", GenerationUsage(1, 7, 3))
    assert calls[0]["stream"] is True and calls[0]["stream_options"] == {"include_usage": True}
    assert stream.closed


async def test_client_reports_an_error_chunk_inside_a_200() -> None:
    error = _Chunk(choices=[], usage=None, model_extra={"error": {"code": 502}})
    client = OpenRouterClient(
        Settings(openrouter_api_key="k"),
        async_client_factory=lambda **_: _client(_Stream([_delta("Hi"), error]), []),
    )

    items = await _collect(client)

    assert items[-1].error == "provider_error" and items[-1].text is None


async def test_client_without_key_makes_no_call() -> None:
    items = await _collect(OpenRouterClient(Settings(openrouter_api_key=None)))

    assert items == [GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")]


def test_stream_answers_in_each_message_language_not_the_first(tmp_path: Path) -> None:
    @dataclass
    class ByWord:
        def detect(self, text: str) -> str:
            return "fr" if "quand" in text.lower() else "en"

    prompts: list[str] = []

    @dataclass
    class Recording(StreamingGenerator):
        async def stream(self, system: str, user: str):
            prompts.append(system)
            async for item in super().stream(system, user):
                yield item

    settings = Settings(data_dir=tmp_path, api_key="api-secret")
    state = ApplicationState(
        settings=settings,
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=FakeRetrieval(),
        generator=Recording(["Friday. [S1]"]),
        language_detector=ByWord(),
    )
    client = TestClient(create_app(settings=settings, state=state))
    first = events(
        client.post(
            "/api/v1/chat",
            json={"message": "Quand ferment les inscriptions au club ?", "stream": True},
            headers=KEY,
        )
    )
    cid = first[0][1]["conversation_id"]
    client.post(
        "/api/v1/chat",
        json={
            "message": "And when does the robotics workshop start this year?",
            "conversation_id": cid,
            "stream": True,
        },
        headers=KEY,
    )

    assert "Answer in French" in prompts[0]
    assert "Answer in English" in prompts[1]
