import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mcclub_rag.ingest import errors as pipeline_errors

logger = logging.getLogger(__name__)

_INGEST_STATUS_CODES = {
    "unsupported_file_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "file_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "url_not_allowed": status.HTTP_400_BAD_REQUEST,
    "fetch_error": status.HTTP_502_BAD_GATEWAY,
    "invalid_input": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "parse_error": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "parse_timeout": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "empty_document": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "index_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "embedding_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "rerank_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


class IngestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _http_error_code(status_code: int) -> str:
    return {
        status.HTTP_401_UNAUTHORIZED: "unauthorized",
        status.HTTP_403_FORBIDDEN: "forbidden",
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limited",
    }.get(status_code, "http_error")


def _message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, Mapping) and isinstance(detail.get("message"), str):
        return detail["message"]
    return "Request failed."


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            str(exc.errors()[0]["msg"]),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(
            exc.status_code,
            _http_error_code(exc.status_code),
            _message(exc.detail),
        )

    @app.exception_handler(IngestError)
    async def ingest_error_handler(_: Request, exc: IngestError) -> JSONResponse:
        status_code = _INGEST_STATUS_CODES.get(exc.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
        return error_response(status_code, exc.code, exc.message)

    # The pipeline raises its own IngestError class (EDGE_CASES "Retrieval errors reach the app
    # as plain 500s"); retrieval failures surface through this handler, not through the adapter.
    @app.exception_handler(pipeline_errors.IngestError)
    async def pipeline_error_handler(_: Request, exc: pipeline_errors.IngestError) -> JSONResponse:
        status_code = _INGEST_STATUS_CODES.get(exc.code, status.HTTP_422_UNPROCESSABLE_CONTENT)
        return error_response(status_code, exc.code, exc.message)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, _: Exception) -> JSONResponse:
        logger.exception("Unhandled request error", extra={"path": request.url.path})
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal",
            "Internal server error.",
        )
