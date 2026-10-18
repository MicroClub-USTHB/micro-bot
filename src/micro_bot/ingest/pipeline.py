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

from micro_bot.ingest.chunk import chunk_document, chunking_signature
from micro_bot.ingest.docstore import DocStore
from micro_bot.ingest.errors import (
    EmptyDocument,
    FileTooLarge,
    IndexStale,
    IngestError,
    InvalidInput,
)
from micro_bot.ingest.identity import clean_topic, doc_id_for, file_topic, point_id, url_topic
from micro_bot.ingest.injection import looks_like_instruction
from micro_bot.ingest.models import (
    Chunk,
    DocumentInfo,
    IngestResult,
    IngestStatus,
    ParsedDocument,
    StoredDocument,
)
from micro_bot.ingest.preprocess import preprocess_file, preprocess_url
from micro_bot.ingest.settings import IngestSettings, get_ingest_settings
from micro_bot.ingest.tokens import TokenCounter, get_token_counter
from micro_bot.retrieval.embed import (
    DenseEncoder,
    SparseEncoder,
    get_dense_encoder,
    get_sparse_encoder,
)
from micro_bot.retrieval.store import DENSE, SPARSE, VectorStore

log = structlog.get_logger(__name__)

# Bump when the point payload or vector layout changes.
# 2: ``instruction_like`` (eval-fixes Req 4) and ``expires_at`` (Req 13)
SCHEMA_VERSION = 2
_REINDEX_COMMAND = "uv run python scripts/reindex.py"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _log_url(url: str) -> str:
    """URL for logs: no credentials, query string or fragment (they may carry tokens)."""
    try:
        return url_topic(url)
    except InvalidInput:
        return "<invalid url>"


def _epoch(moment: datetime | None) -> float | None:
    return None if moment is None else moment.timestamp()


def _payload(
    doc_id: str,
    doc: ParsedDocument,
    chunk: Chunk,
    count: int,
    ingested_at: datetime,
    expires_at: datetime | None = None,
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
        "title": doc.title,
        "source": doc.source,
        "source_type": doc.source_type,
        "ingested_at": ingested_at.isoformat(),
        "instruction_like": looks_like_instruction(chunk.text),
        "expires_at": _epoch(expires_at),  # epoch seconds; retrieval skips it after that
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
        *,
        title: str | None = None,
        topic: str | None = None,
        expires_at: datetime | None = None,
    ) -> IngestResult:
        run = _Run()
        return await self._logged(
            run,
            filename,
            lambda: self._ingest_file(run, path, filename, title, topic, expires_at),
        )

    async def _ingest_file(
        self,
        run: _Run,
        path: str | os.PathLike[str],
        filename: str,
        title: str | None,
        topic: str | None,
        expires_at: datetime | None = None,
    ) -> IngestResult:
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
                run, doc_id, hashlib.sha256(data).hexdigest(), expires_at
            )
            if existing is not None:
                return existing
            run.step("preprocess")
            doc = await self._preprocess_file(data, filename, title=title, settings=self.settings)
            return await self._store_new(run, doc_id, topic, doc, replaces, expires_at)

    async def ingest_url(
        self,
        url: str,
        *,
        title: str | None = None,
        topic: str | None = None,
        expires_at: datetime | None = None,
    ) -> IngestResult:
        run = _Run()
        return await self._logged(
            run,
            _log_url(url),
            lambda: self._ingest_url(run, url, title, topic, expires_at),
        )

    async def _ingest_url(
        self,
        run: _Run,
        url: str,
        title: str | None,
        topic: str | None,
        expires_at: datetime | None = None,
    ) -> IngestResult:
        topic = clean_topic(topic) if topic is not None else None
        # Web pages are hashed on extracted text, so the duplicate check follows the fetch.
        run.step("preprocess")
        doc = await self._preprocess_url(url, title=title, settings=self.settings)
        topic = topic or url_topic(doc.source)  # final URL after redirects
        doc_id = run.doc_id = doc_id_for(topic)
        async with self._writing(doc_id):
            run.step("dedup")
            existing, replaces = await self._check_existing(
                run, doc_id, doc.content_hash, expires_at
            )
            if existing is not None:
                return existing
            return await self._store_new(run, doc_id, topic, doc, replaces, expires_at)

    async def _check_existing(
        self,
        run: _Run,
        doc_id: str,
        content_hash: str,
        expires_at: datetime | None = None,
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
        if marker.expires_at != expires_at:  # re-upload with a new (or no) expiry date
            await self.store.set_fields(doc_id, {"expires_at": _epoch(expires_at)})
            await self.docstore.set_expiry(doc_id, expires_at)
            status = "updated"
        result = IngestResult(
            doc_id=doc_id,
            status=status,
            title=marker.title,
            source=marker.source,
            source_type=marker.source_type,
            language=marker.language,
            chunk_count=marker.chunk_count,
            content_hash=content_hash,
            ocr_used=False,
            duration_ms=run.duration_ms,
        )
        return result, others

    async def _store_new(
        self,
        run: _Run,
        doc_id: str,
        topic: str,
        doc: ParsedDocument,
        replaces: bool,
        expires_at: datetime | None = None,
    ) -> IngestResult:
        """Write the new version, then remove every other one: never zero points."""
        ingested_at = datetime.now(UTC)
        count = await self._index(run, doc_id, doc, ingested_at, expires_at=expires_at)
        run.step("docstore")
        stored = StoredDocument(
            doc_id=doc_id,
            topic=topic,
            ingested_at=ingested_at,
            document=doc,
            expires_at=expires_at,
        )
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
        expires_at: datetime | None = None,
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
                payload=_payload(doc_id, doc, chunk, len(chunks), ingested_at, expires_at),
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
            language=doc.language,
            chunk_count=count,
            content_hash=doc.content_hash,
            ocr_used=doc.ocr_used,
            warnings=doc.warnings,
            duration_ms=run.duration_ms,
        )

    async def list_documents(self) -> list[DocumentInfo]:
        """One entry per complete document, sorted by ``doc_id``."""
        return [
            DocumentInfo(
                doc_id=m.doc_id,
                title=m.title,
                source=m.source,
                source_type=m.source_type,
                language=m.language,
                chunk_count=m.chunk_count,
                content_hash=m.content_hash,
                ingested_at=m.ingested_at,
                expires_at=m.expires_at,
            )
            for m in await self.store.markers()
        ]

    async def delete_document(self, doc_id: str) -> bool:
        """Remove every point and the stored JSON of ``doc_id``; False if it was unknown."""
        async with self._writing(doc_id):
            had_points = await self.store.delete_doc(doc_id)
            had_json = await self.docstore.load(doc_id) is not None
            await self.docstore.delete(doc_id)
        if had_points or had_json:
            log.info("document_deleted", doc_id=doc_id)
        return had_points or had_json

    async def corpus_version(self) -> str:
        """Changes whenever indexed content, expiry or index settings change.

        Read from Qdrant on every call (one marker scroll), so writes made by another process
        (the CLI) are seen at once; nothing is cached in memory.
        """
        # Expired documents drop out, so the version changes when one expires (I14).
        now = datetime.now(UTC)
        documents = [
            [m.doc_id, m.content_hash, _epoch(m.expires_at)]
            for m in await self.store.markers()
            if m.expires_at is None or m.expires_at > now
        ]
        return _digest({"index": self.signature, "documents": documents})

    async def reindex(self) -> None:
        """Rebuild every stored document into a new collection, then swap the alias.

        Search keeps using the old collection until the new one is complete. If any document
        cannot be rebuilt (unreadable JSON, indexed but never stored, or an indexing error),
        the new collection is dropped, the alias stays, and ``IndexStale`` is raised.
        """
        start = time.perf_counter()
        async with self._exclusive():
            stored, failed = await self.docstore.load_all()
            known = {item.doc_id for item in stored}
            failed += [m.doc_id for m in await self.store.markers() if m.doc_id not in known]
            log.info("reindex_started", documents=len(stored), signature=self.signature[:12])
            name = await self.store.next_physical(self.signature)
            physical = await self.store.create_physical(self.signature, name=name)
            chunks = 0
            for item in stored if not failed else ():
                try:
                    chunks += await self._index(
                        _Run(),
                        item.doc_id,
                        item.document,
                        item.ingested_at,
                        collection=physical,
                        expires_at=item.expires_at,
                    )
                except IngestError as exc:
                    log.warning("document_reindex_failed", doc_id=item.doc_id, error_code=exc.code)
                    failed.append(item.doc_id)
            if failed:
                await self.store.drop_physical(physical)
                log.warning("reindex_aborted", failed=sorted(failed))
                raise IndexStale(
                    f"reindex failed for {len(failed)} document(s) "
                    f"({', '.join(sorted(failed)[:5])}); the previous index is still served"
                )
            await self.store.switch_alias(physical)
        log.info(
            "reindex_finished",
            documents=len(stored),
            chunks=chunks,
            duration_ms=round((time.perf_counter() - start) * 1000),
        )

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
    *,
    title: str | None = None,
    topic: str | None = None,
    expires_at: datetime | None = None,
) -> IngestResult:
    pipeline = await get_pipeline()
    return await pipeline.ingest_file(
        path, filename, title=title, topic=topic, expires_at=expires_at
    )


async def ingest_url(
    url: str,
    *,
    title: str | None = None,
    topic: str | None = None,
    expires_at: datetime | None = None,
) -> IngestResult:
    pipeline = await get_pipeline()
    return await pipeline.ingest_url(url, title=title, topic=topic, expires_at=expires_at)


async def list_documents() -> list[DocumentInfo]:
    return await (await get_pipeline()).list_documents()


async def delete_document(doc_id: str) -> bool:
    return await (await get_pipeline()).delete_document(doc_id)


async def corpus_version() -> str:
    return await (await get_pipeline()).corpus_version()
