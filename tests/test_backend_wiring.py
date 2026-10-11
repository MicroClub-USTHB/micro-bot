"""Integration tests for the production backend adapters in ``main.py`` (eeq/wire-backends).

``RealDocumentBackend``, ``RealLanguageDetector``, ``PipelineProbe`` and the lifespan wiring
run against nisbenz's module seam adopted on the fake in-memory pipeline from
``pipeline_helpers``, so no Qdrant server and no model files are needed.
"""

import io
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from mcclub_rag.config import Settings, get_settings
from mcclub_rag.errors import IngestError
from mcclub_rag.ingest import errors as ingest_errors
from mcclub_rag.ingest import pipeline as ingest_pipeline
from mcclub_rag.main import (
    PipelineProbe,
    RealDocumentBackend,
    RealLanguageDetector,
    create_app,
    default_state,
)
from mcclub_rag.retrieval.backend import HybridRetriever
from tests.pipeline_helpers import make_pipeline

GUIDE = "# Guide du club\n\nLes inscriptions ferment vendredi. Le club propose des ateliers."


def upload(name: str, text: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(text.encode()), filename=name)


async def ingest(backend: RealDocumentBackend, name: str, text: str, visibility: str = "public"):
    return await backend.ingest_file(
        upload(name, text), visibility=visibility, topic=None, expires_at=None
    )


@pytest.fixture
async def backend(tmp_path: Path) -> AsyncIterator[RealDocumentBackend]:
    await ingest_pipeline.init_pipeline(pipeline=make_pipeline(tmp_path))
    yield RealDocumentBackend()
    await ingest_pipeline.close_pipeline()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مرحبا بكم في النادي الجامعي", "ar"),
        ("Bonjour, comment fonctionne le club ?", "fr"),
        ("Hello, how does the club work?", "en"),
    ],
)
def test_real_language_detector_delegates_to_detect_language(text: str, expected: str) -> None:
    assert RealLanguageDetector().detect(text) == expected


def test_default_state_wires_the_real_backends(tmp_path: Path) -> None:
    state = default_state(Settings(data_dir=tmp_path))

    assert isinstance(state.documents, RealDocumentBackend)
    assert isinstance(state.document_probe, PipelineProbe)
    assert isinstance(state.language_detector, RealLanguageDetector)


async def test_ingest_file_maps_created_to_indexed(backend: RealDocumentBackend) -> None:
    result = await ingest(backend, "guide.md", GUIDE)

    assert result.status == "indexed"
    assert result.chunk_count >= 1
    assert result.doc_id  # uuid5 of the default topic, derived inside the pipeline
    assert result.error is None


async def test_reingesting_the_same_bytes_reports_duplicate(backend: RealDocumentBackend) -> None:
    first = await ingest(backend, "guide.md", GUIDE)
    second = await ingest(backend, "guide.md", GUIDE)

    assert second.status == "duplicate"
    assert second.doc_id == first.doc_id


async def test_pipeline_ingest_errors_are_translated_to_the_app_ingest_error(
    backend: RealDocumentBackend,
) -> None:
    with pytest.raises(IngestError) as excinfo:
        await ingest(backend, "blob.bin", "junk")

    assert type(excinfo.value) is IngestError  # the app class, not ingest.errors.IngestError
    assert excinfo.value.code == "unsupported_file_type"


async def test_upload_bytes_go_through_a_temp_file_that_is_removed(
    backend: RealDocumentBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    real = ingest_pipeline.ingest_file

    async def spy(path, filename, visibility, **kwargs):
        seen["path"] = Path(path)
        seen["exists_during_call"] = Path(path).is_file()
        return await real(path, filename, visibility, **kwargs)

    monkeypatch.setattr(ingest_pipeline, "ingest_file", spy)

    result = await ingest(backend, "guide.md", GUIDE)

    assert result.status == "indexed"
    assert seen["exists_during_call"] is True
    assert not seen["path"].exists()  # type: ignore[union-attr]


async def test_list_documents_maps_info_fields_and_topic_from_the_docstore(
    backend: RealDocumentBackend,
) -> None:
    await ingest(backend, "guide.md", GUIDE, visibility="members")

    docs = await backend.list_documents()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.topic == "guide"  # default topic: the uploaded filename's stem
    assert doc.status == "indexed"
    assert doc.visibility == "members"
    assert doc.chunk_count >= 1
    assert datetime.fromisoformat(doc.created_at)  # ISO string, not a datetime object


async def test_delete_document_and_corpus_version_track_the_index(
    backend: RealDocumentBackend,
) -> None:
    result = await ingest(backend, "guide.md", GUIDE)
    version = await backend.corpus_version()

    assert isinstance(version, str) and version
    assert await backend.delete_document(result.doc_id) is True
    assert await backend.delete_document(result.doc_id) is False
    assert await backend.corpus_version() != version
    assert await backend.list_documents() == []


async def test_pipeline_probe_reports_the_pipeline_health(
    backend: RealDocumentBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await PipelineProbe().check() is True

    async def broken() -> str:
        raise ingest_errors.IndexUnavailable("qdrant is down")

    monkeypatch.setattr(ingest_pipeline, "corpus_version", broken)
    assert await PipelineProbe().check() is False


class FakeReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9] * len(passages)


async def test_lifespan_wires_the_hybrid_retriever_over_the_shared_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    monkeypatch.setenv("MEMBER_TOKEN", "member-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "mcclub_rag.retrieval.backend.get_reranker", lambda settings: FakeReranker()
    )
    # RequestLoggingMiddleware calls configure_logging(), whose cache_logger_on_first_use=True
    # pins the module loggers in chunk.py / pipeline.py / retrieval/backend.py to the JSON
    # config for the rest of the process (structlog.reset_defaults() cannot unbind them), which
    # would blind the capture_logs() tests of later files. The middleware's logging setup has
    # its own tests; here we only exercise the backend wiring.
    monkeypatch.setattr("mcclub_rag.logging.configure_logging", lambda log_level="INFO": None)
    await ingest_pipeline.init_pipeline(pipeline=make_pipeline(tmp_path))
    try:
        settings = Settings(data_dir=tmp_path, openrouter_api_key="or-key")
        with TestClient(create_app(settings=settings)) as client:
            state = client.app.state.mcclub_state
            assert isinstance(state.documents, RealDocumentBackend)
            assert isinstance(state.retrieval, HybridRetriever)  # wired by the lifespan

            events = "# Événements\n\nLe prochain atelier est samedi au local du club."
            created = client.post(
                "/api/v1/documents",
                files={"file": ("events.md", events.encode(), "text/markdown")},
                data={"visibility": "public"},
                headers={"X-Admin-Key": "admin-secret"},
            )
            assert created.status_code == 200
            doc_id = created.json()["doc_id"]

            listed = client.get("/api/v1/documents", headers={"X-Admin-Key": "admin-secret"})
            assert listed.status_code == 200
            assert listed.json() == [
                {
                    "doc_id": doc_id,
                    "title": "Événements",
                    "visibility": "public",
                    "status": "indexed",
                    "chunk_count": listed.json()[0]["chunk_count"],
                    "created_at": listed.json()[0]["created_at"],
                    "topic": "events",
                    "expires_at": None,
                }
            ]

            health = client.get("/api/v1/health")
            assert health.json() == {"status": "ok", "documents": 1}

            member = {"Authorization": "Bearer member-secret"}
            message = {"message": "Quand a lieu le prochain atelier ?"}
            first = client.post("/api/v1/chat", json=message, headers=member)
            assert first.status_code == 200
            body = first.json()
            assert body["grounded"] is True  # disabled generator -> extractive fallback
            assert body["sources"][0]["doc_id"] == doc_id
            assert body["usage"]["llm_calls"] == 0

            # Same question without a conversation_id hits the exact-match cache.
            second = client.post("/api/v1/chat", json=message, headers=member)
            assert second.status_code == 200
            assert second.json()["answer"] == body["answer"]
            assert second.json()["usage"]["llm_calls"] == 0
    finally:
        await ingest_pipeline.close_pipeline()
        get_settings.cache_clear()
