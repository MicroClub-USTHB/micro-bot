"""Hybrid dense + BM25 search in Qdrant with the visibility filter inside the query (Req 2, 3, 7).

One ``query_points`` request: a dense and a BM25 prefetch fused server-side. The role's filter
is set on each prefetch and on the fused query (the server would push the top-level one down
anyway, measured on 1.19.2; being explicit costs nothing).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.retrieval.settings import RetrievalSettings
from mcclub_rag.retrieval.store import DENSE, SPARSE, guard
from mcclub_rag.schemas import Role

_PAYLOAD = [
    "doc_id",
    "content_hash",
    "index",
    "text",
    "title",
    "heading_path",
    "visibility",
    "ingested_at",
]


@dataclass(frozen=True)
class Candidate:
    point_id: str
    doc_id: str
    content_hash: str
    index: int
    title: str
    heading_path: tuple[str, ...]
    text: str
    visibility: str
    ingested_at: datetime
    fused_score: float
    fused_rank: int

    @classmethod
    def from_point(cls, point: qm.ScoredPoint | qm.Record, rank: int) -> "Candidate":
        p = point.payload or {}
        return cls(
            point_id=str(point.id),
            doc_id=p["doc_id"],
            content_hash=p["content_hash"],
            index=int(p["index"]),
            title=p.get("title", ""),
            heading_path=tuple(p.get("heading_path") or ()),
            text=p["text"],
            visibility=p["visibility"],
            ingested_at=datetime.fromisoformat(p["ingested_at"]),
            fused_score=float(getattr(point, "score", 0.0) or 0.0),
            fused_rank=rank,
        )


def visibility_filter(role: Role) -> qm.Filter:
    """public sees ``public`` only; member sees ``public`` and ``members``."""
    if role == "public":
        match: qm.MatchValue | qm.MatchAny = qm.MatchValue(value="public")
    elif role == "member":
        match = qm.MatchAny(any=["public", "members"])
    else:
        raise ValueError(f"unknown role {role!r}")
    return qm.Filter(must=[qm.FieldCondition(key="visibility", match=match)])


def dedupe(candidates: Sequence[Candidate]) -> list[Candidate]:
    """One candidate per ``(doc_id, index)``: the newest version (tie: better fused rank).

    Two versions are searchable for a moment during a replacement, and a crashed ingest's
    chunks stay until the document is re-ingested. Output keeps the fused order.
    """
    best: dict[tuple[str, int], Candidate] = {}
    for c in candidates:
        key = (c.doc_id, c.index)
        current = best.get(key)
        if current is None or (c.ingested_at, -c.fused_rank) > (
            current.ingested_at,
            -current.fused_rank,
        ):
            best[key] = c
    return sorted(best.values(), key=lambda c: c.fused_rank)


class HybridSearch:
    def __init__(self, client: AsyncQdrantClient, alias: str, settings: RetrievalSettings):
        self.client = client
        self.alias = alias
        self.settings = settings

    def _fusion(self) -> qm.RrfQuery | qm.FusionQuery:
        s = self.settings
        if s.fusion == "dbsf":
            return qm.FusionQuery(fusion=qm.Fusion.DBSF)
        weights = None if s.dense_weight == s.sparse_weight else [s.dense_weight, s.sparse_weight]
        return qm.RrfQuery(rrf=qm.Rrf(k=s.rrf_k, weights=weights))

    async def search(
        self, dense: list[float], sparse: qm.SparseVector, role: Role
    ) -> list[Candidate]:
        """Fused candidates (at most ``top_k``) visible to ``role``, deduplicated."""
        flt = visibility_filter(role)
        limit = self.settings.prefetch_limit
        async with guard("search"):
            response = await self.client.query_points(
                self.alias,
                prefetch=[
                    qm.Prefetch(query=dense, using=DENSE, limit=limit, filter=flt),
                    qm.Prefetch(query=sparse, using=SPARSE, limit=limit, filter=flt),
                ],
                query=self._fusion(),
                query_filter=flt,
                limit=self.settings.top_k,
                with_payload=_PAYLOAD,
                with_vectors=False,
            )
        return dedupe([Candidate.from_point(p, rank) for rank, p in enumerate(response.points)])
