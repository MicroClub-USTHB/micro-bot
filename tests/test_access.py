from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage
from micro_bot.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeGenerationService,
    FakeHealthProbe,
    FakeLanguageDetector,
    FakeRetrievalBackend,
    create_app,
)
from micro_bot.schemas import Chunk, Retrieval


def build_client(tmp_path: Path) -> TestClient:
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        title="Club guide",
        text="The workshop is Saturday.",
        score=0.9,
    )
    state = ApplicationState(
        settings=Settings(data_dir=tmp_path, api_key="api-secret", admin_api_key="admin-secret"),
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
        retrieval=FakeRetrievalBackend(Retrieval(chunks=[chunk], top_score=0.9, decision="answer")),
        language_detector=FakeLanguageDetector(),
        generator=FakeGenerationService(
            GenerationResult("The workshop is Saturday. [doc-1]", GenerationUsage(1))
        ),
    )
    return TestClient(create_app(settings=state.settings, state=state))


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic api-secret"}]
)
def test_chat_and_feedback_need_the_api_key(tmp_path: Path, headers: dict[str, str]) -> None:
    with build_client(tmp_path) as client:
        chat = client.post("/api/v1/chat", json={"message": "When?"}, headers=headers)
        feedback = client.post(
            "/api/v1/feedback", json={"conversation_id": "c_1", "rating": "up"}, headers=headers
        )

    for response in (chat, feedback):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"


def test_chat_with_the_api_key_answers(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "When is the workshop?"},
            headers={"Authorization": "Bearer api-secret"},
        )

    assert response.status_code == 200
    assert response.json()["sources"][0]["doc_id"] == "doc-1"


def test_documents_need_the_admin_key_not_the_api_key(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/documents", headers={"Authorization": "Bearer api-secret"})
        health = client.get("/api/v1/health")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert health.status_code == 200
