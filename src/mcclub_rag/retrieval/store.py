"""Qdrant collection, alias and document-level writes (ingest-pipeline Req 6, 7, 8.3).

The physical collection is named after the index signature and reached through the
``QDRANT_COLLECTION`` alias, so a reindex can build a new collection and swap the alias
atomically. The point with ``index == 0`` of each document version is its *marker*: the
pipeline writes it last, so a marker means the whole version is stored. Listing and the
corpus version read markers only.
"""

import warnings
from collections.abc import AsyncIterator, Sequence
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

    def physical_name(self, signature: str, *, alternate: bool = False) -> str:
        """``alternate``: a forced reindex with unchanged settings builds next to the live one."""
        return f"{self.alias}__{signature[:12]}" + ("_b" if alternate else "")

    async def next_physical(self, signature: str) -> str:
        """Name for a rebuild that never collides with the collection currently served."""
        primary = self.physical_name(signature)
        return (
            self.physical_name(signature, alternate=True)
            if await self._alias_target() == primary
            else primary
        )

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
        names = {self.physical_name(signature), self.physical_name(signature, alternate=True)}
        return "ok" if target in names else "stale"

    async def create_physical(self, signature: str, *, name: str | None = None) -> str:
        """(Re)create a physical collection for ``signature``; returns its name."""
        name = name or self.physical_name(signature)
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

    async def drop_physical(self, name: str) -> None:
        async with _guard("collection drop"):
            if await self.client.collection_exists(name):
                await self.client.delete_collection(name)

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

    async def upsert(self, points: Sequence[qm.PointStruct], *, collection: str | None = None):
        """Write ``points`` in order, in batches, each acknowledged before the next."""
        target = collection or self.alias
        async with _guard("upsert"):
            for start in range(0, len(points), self.batch):
                batch = list(points[start : start + self.batch])
                await self.client.upsert(target, batch, wait=True)

    async def versions(self, doc_id: str) -> list[Marker]:
        """Markers of every complete version of ``doc_id``, newest first."""
        return sorted(
            await self._markers(_match(doc_id=doc_id, index=0)),
            key=lambda m: m.ingested_at,
            reverse=True,
        )

    async def count(self, doc_id: str, content_hash: str) -> int:
        flt = qm.Filter(must=_match(doc_id=doc_id, content_hash=content_hash))
        async with _guard("count"):
            return (await self.client.count(self.alias, count_filter=flt, exact=True)).count

    async def set_visibility(self, doc_id: str, visibility: Visibility) -> None:
        async with _guard("visibility update"):
            await self.client.set_payload(
                self.alias,
                payload={"visibility": visibility},
                points=qm.Filter(must=_match(doc_id=doc_id)),
                wait=True,
            )

    async def _delete(self, flt: qm.Filter, collection: str | None = None) -> None:
        async with _guard("delete"):
            await self.client.delete(
                collection or self.alias, points_selector=qm.FilterSelector(filter=flt), wait=True
            )

    async def delete_version(
        self, doc_id: str, content_hash: str, *, collection: str | None = None
    ) -> None:
        await self._delete(
            qm.Filter(must=_match(doc_id=doc_id, content_hash=content_hash)), collection
        )

    async def delete_other_versions(self, doc_id: str, keep_hash: str) -> None:
        flt = qm.Filter(must=_match(doc_id=doc_id), must_not=_match(content_hash=keep_hash))
        await self._delete(flt)

    async def delete_doc(self, doc_id: str) -> bool:
        """Delete every point of ``doc_id``; False if it had none."""
        flt = qm.Filter(must=_match(doc_id=doc_id))
        async with _guard("count"):
            existed = (await self.client.count(self.alias, count_filter=flt, exact=True)).count > 0
        if existed:
            await self._delete(flt)
        return existed

    async def markers(self) -> list[Marker]:
        """The newest complete version of every document, sorted by ``doc_id``."""
        newest: dict[str, Marker] = {}
        for marker in await self._markers(_match(index=0)):
            current = newest.get(marker.doc_id)
            if current is None or marker.ingested_at > current.ingested_at:
                newest[marker.doc_id] = marker
        return [newest[doc_id] for doc_id in sorted(newest)]

    async def _markers(self, must: list[qm.FieldCondition]) -> list[Marker]:
        found: list[Marker] = []
        offset = None
        async with _guard("scroll"):
            while True:
                points, offset = await self.client.scroll(
                    self.alias,
                    scroll_filter=qm.Filter(must=must),
                    limit=_SCROLL_PAGE,
                    offset=offset,
                    with_payload=_MARKER_FIELDS,
                    with_vectors=False,
                )
                found.extend(Marker.from_payload(p.payload or {}) for p in points)
                if offset is None:
                    return found
