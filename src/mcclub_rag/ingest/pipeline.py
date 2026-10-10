"""Ingest pipeline: file or URL -> preprocess -> chunk -> embed -> Qdrant (ingest-pipeline spec).

Module-level functions are the seam with the API (``ingest_file``, ``ingest_url``,
``list_documents``, ``delete_document``, ``corpus_version``); they share one lazily built
``IngestPipeline``. The API calls ``init_pipeline()`` at startup so missing models or a stale
index fail before the first request.
"""

import asyncio
import hashlib
import json
import os
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.ingest.chunk import chunk_document, chunking_signature
from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.errors import (
    EmptyDocument,
    FileTooLarge,
    IndexStale,
    IngestError,
    InvalidInput,
)
from mcclub_rag.ingest.identity import clean_topic, doc_id_for, file_topic, point_id, url_topic
from mcclub_rag.ingest.models import (
    Chunk,
    IngestResult,
    IngestStatus,
    ParsedDocument,
    StoredDocument,
)
from mcclub_rag.ingest.preprocess import preprocess_file, preprocess_url
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings
from mcclub_rag.ingest.tokens import TokenCounter, get_token_counter
from mcclub_rag.retrieval.embed import (
    DenseEncoder,
    SparseEncoder,
    get_dense_encoder,
    get_sparse_encoder,
)
from mcclub_rag.retrieval.store import DENSE, SPARSE, VectorStore

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 1  # bump when the point payload or vector layout changes
_REINDEX_COMMAND = "uv run python scripts/reindex.py"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_visibility(visibility: str) -> None:
    if visibility not in ("public", "members"):
        raise InvalidInput("visibility must be one of public, members")


def _log_url(url: str) -> str:
    """URL for logs: no credentials, query string or fragment (they may carry tokens)."""
    try:
        return url_topic(url)
    except InvalidInput:
        return "<invalid url>"


def _payload(
    doc_id: str, doc: ParsedDocument, chunk: Chunk, count: int, ingested_at: datetime
) -> dict:
    return {
        "doc_id": doc_id,
        "content_hash": doc.content_hash,
        "index": chunk.index,
        "chunk_count": count,
        "text": chunk.text,
        "heading_path": list(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_index": chunk.section_index,
        "language": chunk.language,
        "token_count": chunk.token_count,
        "visibility": doc.visibility,
        "title": doc.title,
        "source": doc.source,
        "source_type": doc.source_type,
        "ingested_at": ingested_at.isoformat(),
    }


class _Run:
    """Per-document bookkeeping: current stage and per-stage durations for the log event."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.stage = "validate"
        self.doc_id: str | None = None
        self.timings: dict[str, int] = {}
        self._mark = self.start

    def step(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings[f"ms_{self.stage}"] = round((now - self._mark) * 1000)
        self.stage, self._mark = stage, now

    @property
    def duration_ms(self) -> int:
        return round((time.perf_counter() - self.start) * 1000)


class IngestPipeline:
    def __init__(
        self,
        settings: IngestSettings,
        *,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        client: AsyncQdrantClient,
        docstore: DocStore | None = None,
        counter: TokenCounter | None = None,
        preprocess_file=preprocess_file,  # injectable: tests run without tessdata
        preprocess_url=preprocess_url,
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.client = client
        self.counter = counter or get_token_counter(settings)
        self.docstore = docstore or DocStore(settings.data_dir / "documents")
        self.store = VectorStore(
            client, settings.qdrant_collection, dense.dim, batch=settings.upsert_batch
        )
        self._preprocess_file = preprocess_file
        self._preprocess_url = preprocess_url
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._gate = asyncio.Condition()  # ingests share it; reindex takes it exclusively
        self._active = 0
        self._reindexing = False
        self.signature = _digest(
            {
                "schema": SCHEMA_VERSION,
                "dense": dense.fingerprint,
                "sparse": sparse.fingerprint,
                "chunking": chunking_signature(settings, self.counter),
            }
        )

    async def start(self) -> Literal["ok", "created", "reindexed"]:
        """Make sure the alias serves a collection built with the current settings."""
        state = await self.store.ensure(self.signature)
        if state != "stale":
            return state
        if not self.settings.reindex_on_startup:
            raise IndexStale(
                "the index was built with other embedder or chunking settings. "
                f"Run: {_REINDEX_COMMAND}"
            )
        await self.reindex()
        return "reindexed"

    def _lock(self, doc_id: str) -> asyncio.Lock:
        lock = self._locks.get(doc_id)
        if lock is None:
            lock = self._locks[doc_id] = asyncio.Lock()
        return lock

    @asynccontextmanager
    async def _writing(self, doc_id: str) -> AsyncIterator[None]:
        """Shared side of the reindex gate, then the per-document lock, then a concurrency slot."""
        async with self._gate:
            await self._gate.wait_for(lambda: not self._reindexing)
            self._active += 1
        try:
            async with self._lock(doc_id), self._semaphore:
                yield
        finally:
            async with self._gate:
                self._active -= 1
                self._gate.notify_all()

    @asynccontextmanager
    async def _exclusive(self) -> AsyncIterator[None]:
        async with self._gate:
            await self._gate.wait_for(lambda: not self._reindexing)
            self._reindexing = True  # new ingests wait from here on
            await self._gate.wait_for(lambda: self._active == 0)
        try:
            yield
        finally:
            async with self._gate:
                self._reindexing = False
                self._gate.notify_all()

    async def _logged(
        self, run: _Run, source: str, work: Callable[[], Awaitable[IngestResult]]
    ) -> IngestResult:
        try:
            result = await work()
        except IngestError as exc:
            log.warning(
                "document_ingest_failed",
                doc_id=run.doc_id,
                source=source,
                error_code=exc.code,
                stage=run.stage,
                duration_ms=run.duration_ms,
            )
            raise
        run.step("done")
        log.info(
            "document_ingested",
            doc_id=result.doc_id,
            status=result.status,
            chunks=result.chunk_count,
            source=source,
            duration_ms=run.duration_ms,
            **run.timings,
        )
        return result

    async def ingest_file(
        self,
        path: str | os.PathLike[str],
        filename: str,
        visibility: str,
        *,
        title: str | None = None,
        topic: str | None = None,
    ) -> IngestResult:
        run = _Run()
        return await self._logged(
            run, filename, lambda: self._ingest_file(run, path, filename, visibility, title, topic)
        )

    async def _ingest_file(
        self,
        run: _Run,
        path: str | os.PathLike[str],
        filename: str,
        visibility: str,
        title: str | None,
        topic: str | None,
    ) -> IngestResult:
        _check_visibility(visibility)
        topic = clean_topic(topic) if topic is not None else file_topic(filename)
        source = Path(path)
        try:
            size = (await asyncio.to_thread(source.stat)).st_size
        except FileNotFoundError:
            raise InvalidInput(f"file not found: {filename}") from None
        if size > self.settings.max_file_bytes:
            raise FileTooLarge(f"file is larger than {self.settings.max_file_mb} MB")
        data = await asyncio.to_thread(source.read_bytes)
        doc_id = run.doc_id = doc_id_for(topic)
        async with self._writing(doc_id):
            # Files are hashed on raw bytes, so a duplicate is caught before parsing or OCR.
            run.step("dedup")
            existing, replaces = await self._check_existing(
                run, doc_id, hashlib.sha256(data).hexdigest(), visibility
            )
            if existing is not None:
                return existing
            run.step("preprocess")
            doc = await self._preprocess_file(
                data, filename, visibility=visibility, title=title, settings=self.settings
            )
            return await self._store_new(run, doc_id, topic, doc, replaces)

    async def ingest_url(
        self, url: str, visibility: str, *, title: str | None = None, topic: str | None = None
    ) -> IngestResult:
        run = _Run()
        return await self._logged(
            run, _log_url(url), lambda: self._ingest_url(run, url, visibility, title, topic)
        )

    async def _ingest_url(
        self, run: _Run, url: str, visibility: str, title: str | None, topic: str | None
    ) -> IngestResult:
        _check_visibility(visibility)
        topic = clean_topic(topic) if topic is not None else None
        # Web pages are hashed on extracted text, so the duplicate check follows the fetch.
        run.step("preprocess")
        doc = await self._preprocess_url(
            url, visibility=visibility, title=title, settings=self.settings
        )
        topic = topic or url_topic(doc.source)  # final URL after redirects
        doc_id = run.doc_id = doc_id_for(topic)
        async with self._writing(doc_id):
            run.step("dedup")
            existing, replaces = await self._check_existing(
                run, doc_id, doc.content_hash, visibility
            )
            if existing is not None:
                return existing
            return await self._store_new(run, doc_id, topic, doc, replaces)

    async def _check_existing(
        self, run: _Run, doc_id: str, content_hash: str, visibility: str
    ) -> tuple[IngestResult | None, bool]:
        """``duplicate`` / ``updated`` result for a complete copy of this content, else None.

        The flag says whether another complete version exists (the new one then replaces it).
        """
        versions = await self.store.versions(doc_id)
        others = any(m.content_hash != content_hash for m in versions)
        marker = next((m for m in versions if m.content_hash == content_hash), None)
        if marker is None or await self.store.count(doc_id, content_hash) != marker.chunk_count:
            return None, others  # absent, or a crashed half-ingest: rewrite it
        if others:  # leftover of an interrupted replacement
            await self.store.delete_other_versions(doc_id, content_hash)
        status: IngestStatus = "duplicate"
        if marker.visibility != visibility:
            await self.store.set_visibility(doc_id, visibility)
            await self.docstore.set_visibility(doc_id, visibility)
            status = "updated"
        result = IngestResult(
            doc_id=doc_id,
            status=status,
            title=marker.title,
            source=marker.source,
            source_type=marker.source_type,
            visibility=visibility,
            language=marker.language,
            chunk_count=marker.chunk_count,
            content_hash=content_hash,
            ocr_used=False,
            duration_ms=run.duration_ms,
        )
        return result, others

    async def _store_new(
        self, run: _Run, doc_id: str, topic: str, doc: ParsedDocument, replaces: bool
    ) -> IngestResult:
        """Write the new version, then remove every other one: never zero points."""
        ingested_at = datetime.now(UTC)
        count = await self._index(run, doc_id, doc, ingested_at)
        run.step("docstore")
        stored = StoredDocument(doc_id=doc_id, topic=topic, ingested_at=ingested_at, document=doc)
        try:
            await self.docstore.save(stored)
        except OSError:
            await self._discard(doc_id, doc.content_hash)
            raise
        await self.store.delete_other_versions(doc_id, doc.content_hash)
        status: IngestStatus = "replaced" if replaces else "created"
        return self._result(run, doc_id, doc, status, count)

    async def _index(
        self,
        run: _Run,
        doc_id: str,
        doc: ParsedDocument,
        ingested_at: datetime,
        *,
        collection: str | None = None,
    ) -> int:
        """Chunk, encode and store one document version; the marker (index 0) goes last."""
        run.step("chunk")
        chunks = await asyncio.to_thread(
            chunk_document, doc, settings=self.settings, counter=self.counter
        )
        if not chunks:
            raise EmptyDocument("document produced no chunks")
        texts = [chunk.embed_text for chunk in chunks]
        run.step("dense")
        dense = await asyncio.to_thread(self.dense.embed_documents, texts)
        run.step("sparse")
        sparse = await asyncio.to_thread(self.sparse.embed_documents, texts)
        run.step("store")
        points = [
            qm.PointStruct(
                id=point_id(doc_id, doc.content_hash, chunk.index),
                vector={DENSE: dense_vector, SPARSE: sparse_vector},
                payload=_payload(doc_id, doc, chunk, len(chunks), ingested_at),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        try:
            await self.store.upsert(points[1:] + points[:1], collection=collection)
        except IngestError:
            await self._discard(doc_id, doc.content_hash, collection=collection)
            raise
        return len(chunks)

    async def _discard(self, doc_id: str, content_hash: str, *, collection: str | None = None):
        """Best-effort removal of a half-written version; the previous version is untouched."""
        try:
            await self.store.delete_version(doc_id, content_hash, collection=collection)
        except IngestError as exc:
            log.warning("partial_version_cleanup_failed", doc_id=doc_id, error_code=exc.code)

    def _result(
        self, run: _Run, doc_id: str, doc: ParsedDocument, status: IngestStatus, count: int
    ) -> IngestResult:
        return IngestResult(
            doc_id=doc_id,
            status=status,
            title=doc.title,
            source=doc.source,
            source_type=doc.source_type,
            visibility=doc.visibility,
            language=doc.language,
            chunk_count=count,
            content_hash=doc.content_hash,
            ocr_used=doc.ocr_used,
            warnings=doc.warnings,
            duration_ms=run.duration_ms,
        )

    async def reindex(self) -> None:
        raise NotImplementedError  # task 6.8

    async def close(self) -> None:
        await self.client.close()


_pipeline: IngestPipeline | None = None
_init_lock = asyncio.Lock()


def build_pipeline(settings: IngestSettings) -> IngestPipeline:
    """Production wiring: real encoders, remote Qdrant, docstore under ``DATA_DIR``."""
    client = AsyncQdrantClient(
        url=settings.qdrant_url, timeout=int(settings.qdrant_timeout_s), check_compatibility=False
    )
    return IngestPipeline(
        settings,
        dense=get_dense_encoder(settings),
        sparse=get_sparse_encoder(settings),
        client=client,
    )


async def init_pipeline(
    settings: IngestSettings | None = None, *, pipeline: IngestPipeline | None = None
) -> IngestPipeline:
    """Build (or adopt) the shared pipeline and run its startup checks; idempotent."""
    global _pipeline
    async with _init_lock:
        if _pipeline is None:
            candidate = pipeline or build_pipeline(settings or get_ingest_settings())
            state = await candidate.start()
            log.info("ingest_pipeline_ready", index=state, signature=candidate.signature[:12])
            _pipeline = candidate
        return _pipeline


async def get_pipeline() -> IngestPipeline:
    return _pipeline or await init_pipeline()


async def close_pipeline() -> None:
    global _pipeline
    async with _init_lock:
        if _pipeline is not None:
            await _pipeline.close()
            _pipeline = None


async def ingest_file(
    path: str | os.PathLike[str],
    filename: str,
    visibility: str,
    *,
    title: str | None = None,
    topic: str | None = None,
) -> IngestResult:
    pipeline = await get_pipeline()
    return await pipeline.ingest_file(path, filename, visibility, title=title, topic=topic)


async def ingest_url(
    url: str, visibility: str, *, title: str | None = None, topic: str | None = None
) -> IngestResult:
    pipeline = await get_pipeline()
    return await pipeline.ingest_url(url, visibility, title=title, topic=topic)
