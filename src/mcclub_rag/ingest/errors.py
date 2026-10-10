"""Typed ingest errors (preprocessing, chunking, pipeline) with stable codes.

The API layer maps ``code`` to an HTTP status and the JSON error shape; ``message`` is
safe to show to an admin (no stack traces, no document content).
"""

from typing import ClassVar


class IngestError(Exception):
    code: ClassVar[str] = "ingest_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class InvalidInput(IngestError):
    code = "invalid_input"


class FileTooLarge(IngestError):
    code = "file_too_large"


class UnsupportedFileType(IngestError):
    code = "unsupported_file_type"


class ParseError(IngestError):
    code = "parse_error"


class ParseTimeout(ParseError):
    code = "parse_timeout"


class EmptyDocument(IngestError):
    code = "empty_document"


class FetchError(IngestError):
    code = "fetch_error"


class UrlNotAllowed(FetchError):
    code = "url_not_allowed"


class OcrConfigError(IngestError):
    """Raised at startup when OCR language data is missing (not a per-document error)."""

    code = "ocr_config_error"


class TokenizerConfigError(IngestError):
    """Raised at startup when the embedder's tokenizer.json is missing or unreadable."""

    code = "tokenizer_config_error"


class IndexUnavailable(IngestError):
    """Qdrant is unreachable, timed out or rejected a request (suggested HTTP 503)."""

    code = "index_unavailable"


class IndexStale(IngestError):
    """Raised at startup when the index was built with other settings and reindexing is off."""

    code = "index_stale"


class EmbeddingError(IngestError):
    """The embedding model failed at inference time."""

    code = "embedding_error"


class EmbedderConfigError(IngestError):
    """Raised at startup when embedder or BM25 files are missing, corrupt or misconfigured."""

    code = "embedder_config_error"


ALL_ERRORS: tuple[type[IngestError], ...] = (
    InvalidInput,
    FileTooLarge,
    UnsupportedFileType,
    ParseError,
    ParseTimeout,
    EmptyDocument,
    FetchError,
    UrlNotAllowed,
    OcrConfigError,
    TokenizerConfigError,
    IndexUnavailable,
    IndexStale,
    EmbeddingError,
    EmbedderConfigError,
)
