import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from mcclub_rag.config import Settings, get_settings
from mcclub_rag.errors import IngestError
from mcclub_rag.main import (
    ApplicationState,
    FakeDocumentBackend,
    FakeHealthProbe,
    create_app,
)
from mcclub_rag.schemas import IngestResult, Visibility


class ErrorDocumentBackend(FakeDocumentBackend):
    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code

    async def ingest_file(
        self,
        file: UploadFile,
        *,
        visibility: Visibility,
        topic: str | None,
        expires_at: str | None,
    ) -> IngestResult:
        raise IngestError(self.code, "pipeline rejected the upload")


class FailedDocumentBackend(FakeDocumentBackend):
    def __init__(self, error: str | None) -> None:
        super().__init__()
        self.error = error

    async def ingest_file(
        self,
        file: UploadFile,
        *,
        visibility: Visibility,
        topic: str | None,
        expires_at: str | None,
    ) -> IngestResult:
        return IngestResult(
            doc_id="failed-doc",
            title="failed.txt",
            status="failed",
            chunk_count=0,
            error=self.error,
        )


def build_client(
    tmp_path: Path,
    documents: FakeDocumentBackend | None = None,
    *,
    document_healthy: bool = True,
    provider_healthy: bool = True,
) -> TestClient:
    settings = Settings(data_dir=tmp_path, admin_api_key="admin-secret")
    state = ApplicationState(
        settings=settings,
        documents=documents or FakeDocumentBackend(),
        document_probe=FakeHealthProbe(document_healthy),
        provider_probe=FakeHealthProbe(provider_healthy),
    )
    return TestClient(create_app(settings=settings, state=state), raise_server_exceptions=False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[TestClient]:
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    with build_client(tmp_path) as test_client:
        yield test_client
    get_settings.cache_clear()


def admin_headers() -> dict[str, str]:
    return {"X-Admin-Key": "admin-secret"}


def upload(client: TestClient):
    return client.post(
        "/api/v1/documents",
        headers=admin_headers(),
        files={"file": ("guide.txt", b"club guide", "text/plain")},
        data={
            "visibility": "members",
            "topic": "guide",
            "expires_at": "2026-12-31T00:00:00Z",
        },
    )


def test_health_reports_document_count_and_degraded_dependencies(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok", "documents": 0}
    upload(client)
    assert client.get("/api/v1/health").json() == {"status": "ok", "documents": 1}

    client.app.state.mcclub_state.document_probe.healthy = False
    assert client.get("/api/v1/health").json() == {"status": "degraded", "documents": 1}


def test_health_never_fails_when_a_probe_is_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    with build_client(tmp_path, provider_healthy=False) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "documents": 0}
    get_settings.cache_clear()


def test_feedback_appends_jsonl_and_returns_exact_success(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/api/v1/feedback",
        json={"conversation_id": "c_123", "rating": "up", "comment": "Useful"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert json.loads((tmp_path / "feedback.jsonl").read_text().strip()) == {
        "conversation_id": "c_123",
        "rating": "up",
        "comment": "Useful",
    }


def test_feedback_validation_uses_global_json_errors(client: TestClient) -> None:
    response = client.post("/api/v1/feedback", json={"conversation_id": "c_123", "rating": "bad"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_app_routes_return_json_for_not_found_and_method_errors(client: TestClient) -> None:
    for response, status_code, code in [
        (client.get("/api/v1/missing"), 404, "not_found"),
        (client.post("/api/v1/health"), 405, "method_not_allowed"),
    ]:
        assert response.status_code == status_code
        assert set(response.json()) == {"error"}
        assert response.json()["error"]["code"] == code


def test_documents_forward_upload_metadata_list_and_delete(client: TestClient) -> None:
    created = upload(client)
    assert created.status_code == 200
    doc_id = created.json()["doc_id"]

    listed = client.get("/api/v1/documents", headers=admin_headers())
    assert listed.status_code == 200
    assert listed.json() == [
        {
            "doc_id": doc_id,
            "title": "guide.txt",
            "visibility": "members",
            "status": "indexed",
            "chunk_count": 1,
            "created_at": listed.json()[0]["created_at"],
            "topic": "guide",
            "expires_at": "2026-12-31T00:00:00Z",
        }
    ]

    deleted = client.delete(f"/api/v1/documents/{doc_id}", headers=admin_headers())
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True}
    assert client.get("/api/v1/documents", headers=admin_headers()).json() == []


def test_documents_require_admin(client: TestClient) -> None:
    response = client.get("/api/v1/documents")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_duplicate_upload_is_a_200(client: TestClient) -> None:
    first = upload(client)
    second = upload(client)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()


@pytest.mark.parametrize(
    ("code", "status_code"),
    [
        ("unsupported_file_type", 415),
        ("file_too_large", 413),
        ("url_not_allowed", 400),
        ("fetch_error", 502),
        ("invalid_input", 422),
        ("parse_error", 422),
        ("parse_timeout", 422),
        ("empty_document", 422),
        ("index_unavailable", 503),
        ("embedding_error", 500),
        ("rerank_error", 500),
    ],
)
def test_typed_ingest_errors_bubble_to_global_json_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: str, status_code: int
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    with build_client(tmp_path, ErrorDocumentBackend(code)) as client:
        response = upload(client)
    assert response.status_code == status_code
    assert response.json()["error"] == {
        "code": code,
        "message": "pipeline rejected the upload",
    }
    get_settings.cache_clear()


@pytest.mark.parametrize("error", ["parse_error", None])
def test_failed_ingest_result_never_returns_200(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: str | None
) -> None:
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()
    with build_client(tmp_path, FailedDocumentBackend(error)) as client:
        response = upload(client)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == (error or "invalid_input")
    get_settings.cache_clear()


def _put(client: TestClient, name: str, replaces: str, content: bytes, **form: str):
    return client.put(
        "/api/v1/documents",
        headers={"X-Admin-Key": "admin-secret"},
        files={"file": (name, content, "text/plain")},
        data={"replaces": replaces, **form},
    )


def test_put_replaces_the_named_document_under_a_new_file_name(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        headers = {"X-Admin-Key": "admin-secret"}
        first = client.post(
            "/api/v1/documents",
            headers=headers,
            files={"file": ("bylaws.txt", b"fee 500", "text/plain")},
            data={"visibility": "members"},
        )

        replaced = _put(client, "bylaws_2027.txt", "bylaws.txt", b"fee 700")

        assert replaced.status_code == 200
        assert replaced.json() == first.json()
        listing = client.get("/api/v1/documents", headers=headers).json()
        assert [(d["doc_id"], d["title"], d["visibility"]) for d in listing] == [
            (first.json()["doc_id"], "bylaws_2027.txt", "members")
        ]


def test_put_with_an_unknown_name_is_a_404_and_ingests_nothing(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = _put(client, "new.txt", "bylaw", b"content")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        listing = client.get("/api/v1/documents", headers={"X-Admin-Key": "admin-secret"})
        assert listing.json() == []


def test_put_can_change_visibility_and_needs_the_admin_key(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        client.post(
            "/api/v1/documents",
            headers={"X-Admin-Key": "admin-secret"},
            files={"file": ("guide.txt", b"v1", "text/plain")},
            data={"visibility": "members"},
        )

        denied = client.put(
            "/api/v1/documents",
            files={"file": ("guide.txt", b"v2", "text/plain")},
            data={"replaces": "guide"},
        )
        changed = _put(client, "guide.txt", "guide", b"v2", visibility="public")

        assert denied.status_code == 403
        assert changed.status_code == 200
        listing = client.get("/api/v1/documents", headers={"X-Admin-Key": "admin-secret"})
        assert listing.json()[0]["visibility"] == "public"
