import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_LOG_CONTEXT = "_request_log_context"


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=log_level.upper())
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        # Deliberately False: a cached bound logger keeps the processor chain it first bound
        # with (structlog monkey-patches its own lazy proxy), so structlog.testing.capture_logs()
        # and any later reconfigure would never see module-level loggers again. Binding per call
        # is cheap at this app's log volume.
        cache_logger_on_first_use=False,
    )


def set_request_log_context(
    request: Request,
    *,
    role: str | None = None,
    llm_calls: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    corpus_version: str | None = None,
    conversation_id: str | None = None,
    language: str | None = None,
    answer_language: str | None = None,
) -> None:
    context = getattr(request.state, _REQUEST_LOG_CONTEXT, {}).copy()
    for key, value in {
        "role": role,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "corpus_version": corpus_version,
        "conversation_id": conversation_id,
        "language": language,
        "answer_language": answer_language,
    }.items():
        if value is not None:
            context[key] = value
    setattr(request.state, _REQUEST_LOG_CONTEXT, context)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        *,
        log_level: str = "INFO",
        logger: Any | None = None,
    ) -> None:
        super().__init__(app)
        configure_logging(log_level)
        self.logger = logger or structlog.get_logger("micro_bot.request")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started_at = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            context = getattr(request.state, _REQUEST_LOG_CONTEXT, {})
            route = request.scope.get("route")
            event: dict[str, Any] = {
                "route": getattr(route, "path", request.url.path),
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                "llm_calls": context.get("llm_calls", 0),
                "prompt_tokens": context.get("prompt_tokens", 0),
                "completion_tokens": context.get("completion_tokens", 0),
                "corpus_version": context.get("corpus_version"),
            }
            for key in ("conversation_id", "language", "answer_language"):
                if context.get(key) is not None:
                    event[key] = context[key]
            self.logger.info("request_completed", **event)
