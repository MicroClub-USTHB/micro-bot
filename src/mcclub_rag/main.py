from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from fastapi import FastAPI, UploadFile

from mcclub_rag.api.documents import router as documents_router
from mcclub_rag.api.feedback import router as feedback_router
from mcclub_rag.api.health import router as health_router
from mcclub_rag.config import Settings, get_settings
from mcclub_rag.errors import IngestError, install_error_handlers
from mcclub_rag.logging import RequestLoggingMiddleware
from mcclub_rag.schemas import (
    DocInfo,
    IngestResult,
    Visibility,
)


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
class ApplicationState:
    settings: Settings
    documents: DocumentBackend
    document_probe: ReadinessProbe
    provider_probe: ReadinessProbe


def default_state(settings: Settings) -> ApplicationState:
    """Build the temporary local-app fallback until nisbenz injects its backend."""
    return ApplicationState(
        settings=settings,
        documents=FakeDocumentBackend(),
        document_probe=FakeHealthProbe(),
        provider_probe=FakeHealthProbe(),
    )


def create_app(
    *, settings: Settings | None = None, state: ApplicationState | None = None
) -> FastAPI:
    app_settings = settings or get_settings()
    app_state = state or default_state(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app_state.settings.data_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.mcclub_state = app_state
    install_error_handlers(app)
    app.add_middleware(RequestLoggingMiddleware, log_level=app_settings.log_level)
    app.include_router(health_router)
    app.include_router(feedback_router)
    app.include_router(documents_router)
    return app


app = create_app()
