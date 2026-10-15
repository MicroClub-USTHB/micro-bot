import asyncio
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import structlog
from fastapi import FastAPI, UploadFile

from mcclub_rag.api.chat import router as chat_router
from mcclub_rag.api.documents import router as documents_router
from mcclub_rag.api.feedback import router as feedback_router
from mcclub_rag.api.health import router as health_router
from mcclub_rag.cache import ExactMatchCache
from mcclub_rag.config import Settings, get_settings
from mcclub_rag.conversation import ConversationStore
from mcclub_rag.errors import IngestError, install_error_handlers
from mcclub_rag.ingest import errors as ingest_errors
from mcclub_rag.ingest import pipeline as ingest_pipeline
from mcclub_rag.ingest.models import DocumentInfo as IngestDocumentInfo
from mcclub_rag.llm.client import GenerationResult, GenerationUsage, OpenRouterClient
from mcclub_rag.llm.quota import CircuitBreaker, DailyQuota
from mcclub_rag.llm.rewrite import QueryRewrite, QueryRewriter
from mcclub_rag.logging import RequestLoggingMiddleware
from mcclub_rag.rate_limit import SlidingWindowRateLimiter
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.schemas import (
    ChatResponse,
    DocInfo,
    IngestResult,
    Retrieval,
    Role,
    Visibility,
)
from mcclub_rag.text.lang import detect_language

_log = structlog.get_logger(__name__)


class DocumentBackend(Protocol):
    async def ingest_file(
        self,
        file: UploadFile,
        *,
        visibility: Visibility,
        topic: str | None,
        expires_at: str | None,
    ) -> IngestResult: ...

    async def list_documents(self) -> list[DocInfo]: ...

    async def delete_document(self, doc_id: str) -> bool: ...

    async def corpus_version(self) -> str: ...


class ReadinessProbe(Protocol):
    async def check(self) -> bool: ...


class RetrievalBackend(Protocol):
    async def retrieve(
        self, query: str, role: Role, history: list[object], *, rewrite: QueryRewrite | None = None
    ) -> Retrieval: ...


class LanguageDetector(Protocol):
    def detect(self, text: str) -> str: ...


class GenerationService(Protocol):
    async def generate(self, system: str, user: str) -> GenerationResult: ...


@dataclass
class FakeHealthProbe:
    healthy: bool = True

    async def check(self) -> bool:
        return self.healthy


@dataclass
class FakeDocumentBackend:
    documents: dict[str, DocInfo] = field(default_factory=dict)
    _content_hashes: dict[str, str] = field(default_factory=dict)
    _corpus_version: int = 0

    async def ingest_file(
        self,
        file: UploadFile,
        *,
        visibility: Visibility,
        topic: str | None,
        expires_at: str | None,
    ) -> IngestResult:
        content = await file.read()
        if not content:
            raise IngestError("empty_document", "The uploaded document is empty.")

        title = file.filename or "document"
        document_topic = topic or Path(title).stem
        doc_id = str(uuid5(NAMESPACE_URL, document_topic))
        content_hash = sha256(content).hexdigest()
        if self._content_hashes.get(doc_id) == content_hash:
            return IngestResult(
                doc_id=doc_id,
                title=title,
                status="duplicate",
                chunk_count=0,
            )

        self.documents[doc_id] = DocInfo(
            doc_id=doc_id,
            title=title,
            visibility=visibility,
            status="indexed",
            chunk_count=1,
            created_at=datetime.now(UTC).isoformat(),
            topic=document_topic,
            expires_at=expires_at,
        )
        self._content_hashes[doc_id] = content_hash
        self._corpus_version += 1
        return IngestResult(doc_id=doc_id, title=title, status="indexed", chunk_count=1)

    async def list_documents(self) -> list[DocInfo]:
        return list(self.documents.values())

    async def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        del self._content_hashes[doc_id]
        self._corpus_version += 1
        return True

    async def corpus_version(self) -> str:
        return str(self._corpus_version)


@dataclass
class FakeRetrievalBackend:
    retrieval: Retrieval = field(
        default_factory=lambda: Retrieval(chunks=[], top_score=0.0, decision="refuse")
    )

    async def retrieve(
        self, query: str, role: Role, history: list[object], *, rewrite: QueryRewrite | None = None
    ) -> Retrieval:
        return self.retrieval


@dataclass
class FakeLanguageDetector:
    language: str = "en"

    def detect(self, text: str) -> str:
        return self.language


@dataclass
class FakeGenerationService:
    result: GenerationResult = field(
        default_factory=lambda: GenerationResult(None, GenerationUsage(llm_calls=0), "disabled")
    )

    async def generate(self, system: str, user: str) -> GenerationResult:
        return self.result


@dataclass
class OpenRouterGenerationService:
    """Async adapter over the sync ``OpenRouterClient``: one provider call, off the event loop."""

    client: OpenRouterClient

    async def generate(self, system: str, user: str) -> GenerationResult:
        return await asyncio.to_thread(self.client.generate, system, user)


_INGEST_STATUS_MAP = {
    "created": "indexed",
    "replaced": "indexed",
    "updated": "indexed",
    "duplicate": "duplicate",
}


class RealDocumentBackend:
    """Thin adapter over nisbenz's ingest pipeline module functions."""

    async def ingest_file(
        self,
        file: UploadFile,
        *,
        visibility: Visibility,
        topic: str | None,
        expires_at: str | None,
    ) -> IngestResult:
        expiry = parse_expiry(expires_at)  # before reading the upload: a bad date is a 422
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            result = await _run_ingest(
                ingest_pipeline.ingest_file,
                tmp_path,
                file.filename or "document",
                visibility,
                topic=topic,
                expires_at=expiry,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return _map_ingest_result(result)

    async def list_documents(self) -> list[DocInfo]:
        docs = await _run_ingest(ingest_pipeline.list_documents)
        # Topics live only in the docstore JSON; the pipeline's markers don't carry them.
        pipeline = await ingest_pipeline.get_pipeline()
        stored, _ = await pipeline.docstore.load_all()
        topics = {item.doc_id: item.topic for item in stored}
        return [_map_doc_info(d, topics.get(d.doc_id)) for d in docs]

    async def delete_document(self, doc_id: str) -> bool:
        return await _run_ingest(ingest_pipeline.delete_document, doc_id)

    async def corpus_version(self) -> str:
        return await _run_ingest(ingest_pipeline.corpus_version)


class RealLanguageDetector:
    def detect(self, text: str) -> str:
        return detect_language(text)


@dataclass
class PipelineProbe:
    async def check(self) -> bool:
        try:
            await ingest_pipeline.corpus_version()
            return True
        except Exception:
            return False


@dataclass
class ProviderProbe:
    settings: Settings

    async def check(self) -> bool:
        return bool(self.settings.openrouter_api_key)


def _map_ingest_result(result: object) -> IngestResult:
    status = _INGEST_STATUS_MAP.get(getattr(result, "status", "created"), "indexed")
    return IngestResult(
        doc_id=result.doc_id,
        title=result.title,
        status=status,
        chunk_count=result.chunk_count,
    )


def parse_expiry(value: str | None) -> datetime | None:
    """``expires_at`` form field: ISO 8601 date or datetime; a date ends at its last second UTC,
    a time without an offset is UTC. Blank = no expiry."""
    if value is None or not value.strip():
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        raise IngestError(
            "invalid_input", "expires_at must be an ISO 8601 date or datetime"
        ) from None
    if len(value.strip()) == 10:  # a bare date: valid until the end of that day
        moment = moment.replace(hour=23, minute=59, second=59)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _map_doc_info(doc: IngestDocumentInfo, topic: str | None = None) -> DocInfo:
    return DocInfo(
        doc_id=doc.doc_id,
        title=doc.title,
        visibility=doc.visibility,
        status="indexed",
        chunk_count=doc.chunk_count,
        created_at=doc.ingested_at.isoformat(),
        topic=topic,
        expires_at=doc.expires_at.isoformat() if doc.expires_at else None,
    )


async def _run_ingest(
    func: Callable[..., Awaitable[object]], *args: object, **kwargs: object
) -> object:
    try:
        return await func(*args, **kwargs)
    except ingest_errors.IngestError as exc:
        raise IngestError(exc.code, exc.message) from exc


@dataclass
class ApplicationState:
    settings: Settings
    documents: DocumentBackend
    document_probe: ReadinessProbe
    provider_probe: ReadinessProbe
    retrieval: RetrievalBackend = field(default_factory=FakeRetrievalBackend)
    language_detector: LanguageDetector = field(default_factory=FakeLanguageDetector)
    generator: GenerationService = field(default_factory=FakeGenerationService)
    rewriter: QueryRewriter | None = None  # the optional first LLM call (query-rewrite)
    conversations: ConversationStore = field(default_factory=ConversationStore)
    rate_limiter: SlidingWindowRateLimiter | None = None
    chat_cache: ExactMatchCache[ChatResponse] | None = None
    quota: DailyQuota | None = None
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    corpus_version: str = "0"

    def __post_init__(self) -> None:
        if self.rate_limiter is None:
            self.rate_limiter = SlidingWindowRateLimiter(
                self.settings.chat_rate_limit_requests,
                self.settings.chat_rate_limit_window_s,
            )
        if self.chat_cache is None:
            self.chat_cache = ExactMatchCache(lambda: self.corpus_version)
        if self.quota is None:
            self.quota = DailyQuota(
                self.settings.data_dir,
                self.settings.llm_daily_limit,
                self.settings.llm_reserve,
            )


def default_state(settings: Settings) -> ApplicationState:
    """Production state wired to nisbenz's real ingestion and retrieval backends."""
    generator = _generator(settings)
    rewriter = None
    if settings.rewrite_enabled and settings.openrouter_api_key:
        rewriter = QueryRewriter(
            timeout_s=settings.rewrite_timeout_s, cache_size=settings.rewrite_cache_size
        )
    return ApplicationState(
        settings=settings,
        documents=RealDocumentBackend(),
        document_probe=PipelineProbe(),
        provider_probe=ProviderProbe(settings),
        language_detector=RealLanguageDetector(),
        generator=generator,
        rewriter=rewriter,
    )


def _generator(settings: Settings) -> GenerationService:
    if settings.openrouter_api_key:
        return OpenRouterGenerationService(OpenRouterClient(settings))
    _log.warning("llm_disabled", reason="OPENROUTER_API_KEY not set: extractive fallback only")
    return FakeGenerationService()


def create_app(
    *, settings: Settings | None = None, state: ApplicationState | None = None
) -> FastAPI:
    app_settings = settings or get_settings()
    app_state = state or default_state(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app_state.settings.data_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(app_state.documents, RealDocumentBackend):
            pipeline = await ingest_pipeline.init_pipeline()
            app_state.retrieval = HybridRetriever.from_pipeline(pipeline)
        try:
            yield
        finally:
            if isinstance(app_state.documents, RealDocumentBackend):
                await ingest_pipeline.close_pipeline()

    app = FastAPI(lifespan=lifespan)
    app.state.mcclub_state = app_state
    install_error_handlers(app)
    app.add_middleware(RequestLoggingMiddleware, log_level=app_settings.log_level)
    app.include_router(health_router)
    app.include_router(feedback_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    return app


app = create_app()
