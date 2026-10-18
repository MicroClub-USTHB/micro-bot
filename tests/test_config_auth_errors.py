from collections.abc import Generator

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from micro_bot.config import Settings, get_settings
from micro_bot.deps import require_admin, require_api_key
from micro_bot.errors import IngestError, install_error_handlers


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("API_KEY", "api-secret")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    get_settings.cache_clear()

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/chat")
    async def chat(_: None = Depends(require_api_key)) -> dict[str, bool]:
        return {"ok": True}

    @app.get("/admin")
    async def admin(_: None = Depends(require_admin)) -> dict[str, bool]:
        return {"ok": True}

    @app.get("/ingest/{code}")
    async def ingest_error(code: str) -> None:
        raise IngestError(code, "bad upload")

    @app.get("/http-error")
    async def http_error() -> None:
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/crash")
    async def crash() -> None:
        raise RuntimeError("private traceback")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    get_settings.cache_clear()


def assert_error(response: object, status_code: int, code: str) -> None:
    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert isinstance(response.json()["error"]["message"], str)


def test_settings_loads_documented_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    monkeypatch.setenv("GATE_THRESHOLD", "")
    settings = Settings()

    assert settings.openrouter_model == "google/gemma-4-31b-it:free"
    assert settings.gate_threshold is None
    assert settings.chat_rate_limit_requests == 20
    assert settings.chat_rate_limit_window_s == 60


def test_settings_rejects_invalid_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_RATE_LIMIT_REQUESTS", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_missing_tokens_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/chat")
    async def chat(_: None = Depends(require_api_key)) -> dict[str, bool]:
        return {"ok": True}

    @app.get("/admin")
    async def admin(_: None = Depends(require_admin)) -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as test_client:
        assert_error(test_client.get("/chat"), 401, "unauthorized")
        assert_error(
            test_client.get("/chat", headers={"Authorization": "Bearer any"}),
            401,
            "unauthorized",
        )
        assert_error(test_client.get("/admin"), 403, "forbidden")


def test_api_key_and_admin_auth(client: TestClient) -> None:
    assert_error(client.get("/chat"), 401, "unauthorized")
    assert client.get("/chat", headers={"Authorization": "Bearer api-secret"}).json() == {
        "ok": True
    }
    assert_error(
        client.get("/chat", headers={"Authorization": "Bearer wrong"}),
        401,
        "unauthorized",
    )
    assert_error(
        client.get("/chat", headers={"Authorization": "Basic api-secret"}),
        401,
        "unauthorized",
    )
    assert_error(client.get("/admin"), 403, "forbidden")
    assert_error(client.get("/admin", headers={"X-Admin-Key": "wrong"}), 403, "forbidden")
    assert client.get("/admin", headers={"X-Admin-Key": "admin-secret"}).json() == {"ok": True}


@pytest.mark.parametrize(
    ("ingest_code", "status_code"),
    [
        ("unsupported_file_type", 415),
        ("file_too_large", 413),
        ("url_not_allowed", 400),
        ("fetch_error", 502),
        ("invalid_input", 422),
        ("parse_error", 422),
        ("parse_timeout", 422),
        ("empty_document", 422),
    ],
)
def test_ingest_errors_have_stable_json_mapping(
    client: TestClient, ingest_code: str, status_code: int
) -> None:
    assert_error(client.get(f"/ingest/{ingest_code}"), status_code, ingest_code)


def test_every_standard_error_is_json(client: TestClient) -> None:
    assert_error(client.get("/missing"), 404, "not_found")
    assert_error(client.post("/chat"), 405, "method_not_allowed")
    assert_error(client.get("/http-error"), 418, "http_error")
    assert_error(client.get("/crash"), 500, "internal")


def test_app_defaults_match_the_retrieval_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from micro_bot.retrieval.settings import RetrievalSettings

    for name in ("RERANK_MODEL", "RETRIEVE_TOP_K", "RERANK_TOP_N", "GATE_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    app, retrieval = Settings(_env_file=None), RetrievalSettings()
    assert app.rerank_model == retrieval.rerank_model
    assert app.retrieve_top_k == retrieval.top_k and app.rerank_top_n == retrieval.top_n
    assert app.gate_threshold == retrieval.gate_threshold
