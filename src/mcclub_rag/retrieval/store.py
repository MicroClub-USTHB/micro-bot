"""Qdrant collection, alias and document-level writes (ingest-pipeline Req 6, 7, 8.3).

The physical collection is named after the index signature and reached through the
``QDRANT_COLLECTION`` alias, so a reindex can build a new collection and swap the alias
atomically. The point with ``index == 0`` of each document version is its *marker*: the
pipeline writes it last, so a marker means the whole version is stored. Listing and the
corpus version read markers only.
"""

import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import ApiException

from mcclub_rag.ingest.errors import IndexStale, IndexUnavailable
from mcclub_rag.ingest.models import SourceType, Visibility

DENSE = "dense"
SPARSE = "bm25"
_KEYWORD_FIELDS = ("doc_id", "content_hash", "visibility")
_MARKER_FIELDS = [
    "doc_id",
    "content_hash",
    "visibility",
    "chunk_count",
    "title",
    "source",
    "source_type",
    "language",
    "ingested_at",
]
_SCROLL_PAGE = 256


@dataclass(frozen=True)
class Marker:
    """Document-level payload of one complete version (its ``index == 0`` point)."""

    doc_id: str
    content_hash: str
    visibility: Visibility
    chunk_count: int
    title: str
    source: str
    source_type: SourceType
    language: str
    ingested_at: datetime

    @classmethod
    def from_payload(cls, payload: dict) -> "Marker":
        fields = {name: payload[name] for name in _MARKER_FIELDS}
        fields["ingested_at"] = datetime.fromisoformat(payload["ingested_at"])
        return cls(**fields)


@asynccontextmanager
async def _guard(action: str) -> AsyncIterator[None]:
    try:
        yield
    except (ApiException, httpx.HTTPError, TimeoutError) as exc:
        raise IndexUnavailable(f"vector store unavailable during {action}") from exc


def _match(**fields: str | int) -> list[qm.FieldCondition]:
    return [qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in fields.items()]


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, alias: str, dim: int, *, batch: int = 64):
        self.client = client
        self.alias = alias
        self.dim = dim
        self.batch = batch

    def physical_name(self, signature: str) -> str:
        return f"{self.alias}__{signature[:12]}"

    async def _alias_target(self) -> str | None:
        async with _guard("alias lookup"):
            aliases = (await self.client.get_aliases()).aliases
        return next((a.collection_name for a in aliases if a.alias_name == self.alias), None)

    async def ensure(self, signature: str) -> Literal["ok", "created", "stale"]:
        """Create the collection and alias if missing; report whether they match ``signature``."""
        target = await self._alias_target()
        if target is None:
            async with _guard("collection check"):
                clash = await self.client.collection_exists(self.alias)
            if clash:
                raise IndexStale(
                    f"a collection named {self.alias!r} exists but is not an alias; "
                    "rename or drop it so the pipeline can manage the alias"
                )
            await self.switch_alias(await self.create_physical(signature))
            return "created"
        return "ok" if target == self.physical_name(signature) else "stale"

    async def create_physical(self, signature: str) -> str:
        """(Re)create the physical collection for ``signature``; returns its name."""
        name = self.physical_name(signature)
        async with _guard("collection create"):
            if await self.client.collection_exists(name):
                await self.client.delete_collection(name)
            await self.client.create_collection(
                name,
                vectors_config={DENSE: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            with warnings.catch_warnings():  # local (test) mode ignores payload indexes
                warnings.simplefilter("ignore", UserWarning)
                for field in _KEYWORD_FIELDS:
                    await self.client.create_payload_index(
                        name, field, qm.PayloadSchemaType.KEYWORD
                    )
                await self.client.create_payload_index(name, "index", qm.PayloadSchemaType.INTEGER)
        return name

    async def switch_alias(self, physical: str) -> None:
        """Point the alias at ``physical`` in one atomic operation, then drop the old target."""
        old = await self._alias_target()
        operations: list[qm.AliasOperations] = []
        if old is not None:
            operations.append(
                qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=self.alias))
            )
        operations.append(
            qm.CreateAliasOperation(
                create_alias=qm.CreateAlias(collection_name=physical, alias_name=self.alias)
            )
        )
        async with _guard("alias switch"):
            await self.client.update_collection_aliases(change_aliases_operations=operations)
            if old is not None and old != physical:
                await self.client.delete_collection(old)
