"""Ingest pipeline: file or URL -> preprocess -> chunk -> embed -> Qdrant (ingest-pipeline spec).

Module-level functions are the seam with the API (``ingest_file``, ``ingest_url``,
``list_documents``, ``delete_document``, ``corpus_version``); they share one lazily built
``IngestPipeline``. The API calls ``init_pipeline()`` at startup so missing models or a stale
index fail before the first request.
"""

import asyncio
import hashlib
import json
from typing import Literal

import structlog
from qdrant_client import AsyncQdrantClient

from mcclub_rag.ingest.chunk import chunking_signature
from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.errors import IndexStale
from mcclub_rag.ingest.preprocess import preprocess_file, preprocess_url
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings
from mcclub_rag.ingest.tokens import TokenCounter, get_token_counter
from mcclub_rag.retrieval.embed import (
    DenseEncoder,
    SparseEncoder,
    get_dense_encoder,
    get_sparse_encoder,
)
from mcclub_rag.retrieval.store import VectorStore

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 1  # bump when the point payload or vector layout changes
_REINDEX_COMMAND = "uv run python scripts/reindex.py"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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
