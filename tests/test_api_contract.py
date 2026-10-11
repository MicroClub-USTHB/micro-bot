from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mcclub_rag.config import Settings, get_settings
from mcclub_rag.llm.client import GenerationResult, GenerationUsage
from mcclub_rag.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeHealthProbe,
    FakeLanguageDetector,
    LanguageDetector,
    create_app,
)
from mcclub_rag.schemas import Chunk, Retrieval, Role


@dataclass
class FakeRetrieval:
    result: Retrieval
    calls: int = 0
    roles: list[Role] = field(default_factory=list)

    async def retrieve(self, query: str, role: Role, history: object) -> Retrieval:
        self.calls += 1
        self.roles.append(role)
        return self.result


@dataclass
class FakeGenerator:
    result: GenerationResult
    calls: int = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        return self.result


@dataclass
class RecordingDetector:
    language: str = "fr"
    calls: int = 0

    def detect(self, text: str) -> str:
        self.calls += 1
        return self.language


def chunk(doc_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        doc_id=doc_id,
        title="Club guide",
        text="Registration closes Friday.",
        score=0.9,
    )


def build_client(
    tmp_path: Path,
    *,
    retrieval: FakeRetrieval | None = None,
    generator: FakeGenerator | None = None,
    detector: LanguageDetector | None = None,
    rate_limit: int = 20,
) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        member_token="member-secret",
        chat_rate_limit_requests=rate_limit,
    )
    state = ApplicationState(
        settings=settings,
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=retrieval
        or FakeRetrieval(Retrieval(chunks=[chunk()], top_score=0.9, decision="answer")),
        generator=generator
        or FakeGenerator(
            GenerationResult("Registration closes Friday. [doc-1]", GenerationUsage(1, 3, 4))
        ),
        language_detector=detector or FakeLanguageDetector(),
    )
    return TestClient(create_app(settings=settings, state=state), raise_server_exceptions=False)


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient]:
    get_settings.cache_clear()
    with build_client(tmp_path) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_chat_response_has_the_exact_contract_shape_and_integer_latency(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"message": "When does registration close?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "answer",
        "conversation_id",
        "grounded",
        "sources",
        "usage",
        "latency_ms",
    }
    assert body["conversation_id"].startswith("c_")
    assert body["grounded"] is True
    assert body["sources"] == [
        {"doc_id": "doc-1", "title": "Club guide", "snippet": "Registration closes Friday."}
    ]
    assert body["usage"] == {"llm_calls": 1, "prompt_tokens": 3, "completion_tokens": 4}
    assert type(body["latency_ms"]) is int


def test_long_chat_message_uses_the_global_json_validation_envelope(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"message": "x" * 2001})

    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "validation_error"


def test_rate_limit_runs_before_retrieval_or_generation(tmp_path: Path) -> None:
    retrieval = FakeRetrieval(Retrieval(chunks=[chunk()], top_score=0.9, decision="answer"))
    generator = FakeGenerator(GenerationResult("Answer [doc-1]", GenerationUsage(1)))
    with build_client(tmp_path, retrieval=retrieval, generator=generator, rate_limit=1) as client:
        assert client.post("/api/v1/chat", json={"message": "first"}).status_code == 200
        response = client.post("/api/v1/chat", json={"message": "second"})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"
    assert retrieval.calls == 1
    assert generator.calls == 1


def test_retrieval_refusal_is_ungrounded_and_never_calls_generator(tmp_path: Path) -> None:
    retrieval = FakeRetrieval(Retrieval(chunks=[], top_score=0.0, decision="refuse"))
    generator = FakeGenerator(GenerationResult("unexpected", GenerationUsage(1)))
    with build_client(tmp_path, retrieval=retrieval, generator=generator) as client:
        response = client.post("/api/v1/chat", json={"message": "What is unknown?"})

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["usage"]["llm_calls"] == 0
    assert generator.calls == 0


def test_exact_cache_hit_skips_retrieval_and_generation(tmp_path: Path) -> None:
    retrieval = FakeRetrieval(Retrieval(chunks=[chunk()], top_score=0.9, decision="answer"))
    generator = FakeGenerator(GenerationResult("Answer [doc-1]", GenerationUsage(1)))
    with build_client(tmp_path, retrieval=retrieval, generator=generator) as client:
        first = client.post(
            "/api/v1/chat", json={"message": "Hello, when does registration close?"}
        )
        second = client.post("/api/v1/chat", json={"message": "When does registration close?"})

    assert first.status_code == second.status_code == 200
    assert second.json()["usage"]["llm_calls"] == 0
    assert retrieval.calls == 1
    assert generator.calls == 1


def test_short_follow_up_reuses_stored_conversation_language(tmp_path: Path) -> None:
    detector = RecordingDetector()
    retrieval = FakeRetrieval(Retrieval(chunks=[chunk()], top_score=0.9, decision="answer"))
    generator = FakeGenerator(GenerationResult("Réponse [doc-1]", GenerationUsage(1)))
    with build_client(
        tmp_path, retrieval=retrieval, generator=generator, detector=detector
    ) as client:
        first = client.post("/api/v1/chat", json={"message": "Quand est l atelier?"})
        conversation_id = first.json()["conversation_id"]
        second = client.post(
            "/api/v1/chat", json={"message": "et quand?", "conversation_id": conversation_id}
        )

    assert second.status_code == 200
    assert detector.calls == 1
    assert retrieval.calls == 2


def test_invalid_answer_citations_are_not_grounded(tmp_path: Path) -> None:
    generator = FakeGenerator(GenerationResult("Made up [not-a-doc]", GenerationUsage(1)))
    with build_client(tmp_path, generator=generator) as client:
        response = client.post("/api/v1/chat", json={"message": "Tell me"})

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert response.json()["sources"] == []
