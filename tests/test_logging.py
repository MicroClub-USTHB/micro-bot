import json
import logging

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from mcclub_rag.errors import install_error_handlers
from mcclub_rag.logging import RequestLoggingMiddleware, configure_logging, set_request_log_context


def request_events(caplog: object) -> list[dict[str, object]]:
    records = [record for record in caplog.records if record.name == "mcclub_rag.request"]
    return [json.loads(record.getMessage()) for record in records]


def test_request_logging_emits_one_json_line_with_available_context(caplog: object) -> None:
    configure_logging()
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/api/v1/chat")
    async def chat(request: Request) -> dict[str, bool]:
        set_request_log_context(
            request,
            role="member",
            llm_calls=1,
            prompt_tokens=13,
            completion_tokens=8,
            corpus_version="42",
            conversation_id="c_123",
        )
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="mcclub_rag.request"):
        with TestClient(app) as client:
            assert client.get("/api/v1/chat").json() == {"ok": True}

    events = request_events(caplog)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "request_completed"
    assert event["route"] == "/api/v1/chat"
    assert event["role"] == "member"
    assert isinstance(event["latency_ms"], int)
    assert event["llm_calls"] == 1
    assert event["prompt_tokens"] == 13
    assert event["completion_tokens"] == 8
    assert event["corpus_version"] == "42"
    assert event["conversation_id"] == "c_123"


def test_request_logging_emits_one_line_for_a_handled_exception(caplog: object) -> None:
    configure_logging()
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/api/v1/crash")
    async def crash() -> None:
        raise RuntimeError("private failure")

    with caplog.at_level(logging.INFO, logger="mcclub_rag.request"):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/crash")

    assert response.status_code == 500
    events = request_events(caplog)
    assert len(events) == 1
    assert events[0]["route"] == "/api/v1/crash"
    assert events[0]["role"] == "public"
    assert isinstance(events[0]["latency_ms"], int)
    assert events[0]["llm_calls"] == 0
    assert events[0]["prompt_tokens"] == 0
    assert events[0]["completion_tokens"] == 0
    assert events[0]["corpus_version"] is None


def test_request_logging_uses_standard_defaults(caplog: object) -> None:
    configure_logging()
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    with caplog.at_level(logging.INFO, logger="mcclub_rag.request"):
        with TestClient(app) as client:
            assert client.get("/api/v1/health").json() == {"status": "ok"}

    events = request_events(caplog)
    assert len(events) == 1
    event = events[0]
    assert event["route"] == "/api/v1/health"
    assert event["role"] == "public"
    assert event["llm_calls"] == 0
    assert event["prompt_tokens"] == 0
    assert event["completion_tokens"] == 0
    assert event["corpus_version"] is None
    assert "conversation_id" not in event
