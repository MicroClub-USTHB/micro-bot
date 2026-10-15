"""Hybrid dense + BM25 search in Qdrant with the visibility filter inside the query (Req 2, 3, 7).

One ``query_points`` request: a dense and a BM25 prefetch fused server-side. The role's filter
is set on each prefetch and on the fused query (the server would push the top-level one down
anyway, measured on 1.19.2; being explicit costs nothing).
"""

import time
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
    "instruction_like",
    "language",
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
    instruction_like: bool = False  # payload schema 2; absent on older points
    language: str = "und"  # the chunk's language: picks the rewrite text the reranker sees

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
            instruction_like=bool(p.get("instruction_like", False)),
            language=p.get("language") or "und",
        )


def visibility_filter(role: Role, now: float | None = None) -> qm.Filter:
    """public sees ``public`` only; member sees ``public`` and ``members``. Either way, only
    documents without ``expires_at`` or not yet expired (eval-fixes Req 13)."""
    if role == "public":
        match: qm.MatchValue | qm.MatchAny = qm.MatchValue(value="public")
    elif role == "member":
        match = qm.MatchAny(any=["public", "members"])
    else:
        raise ValueError(f"unknown role {role!r}")
    current = time.time() if now is None else now
    not_expired = qm.Filter(
        should=[
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key="expires_at")),
            qm.IsNullCondition(is_null=qm.PayloadField(key="expires_at")),
            qm.FieldCondition(key="expires_at", range=qm.Range(gt=current)),
        ]
    )
    return qm.Filter(must=[qm.FieldCondition(key="visibility", match=match), not_expired])


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

    def _fusion(self, arms: int = 2) -> qm.RrfQuery | qm.FusionQuery:
        s = self.settings
        if s.fusion == "dbsf":
            return qm.FusionQuery(fusion=qm.Fusion.DBSF)
        weights = None
        if s.dense_weight != s.sparse_weight:  # every arm after the BM25 one is a dense arm
            weights = [s.dense_weight, s.sparse_weight] + [s.dense_weight] * (arms - 2)
        return qm.RrfQuery(rrf=qm.Rrf(k=s.rrf_k, weights=weights))

    async def search(
        self,
        dense: list[float],
        sparse: qm.SparseVector,
        role: Role,
        *,
        query_language: str | None = None,
        extra_dense: Sequence[tuple[list[float], str | None]] = (),
    ) -> list[Candidate]:
        """Fused candidates (at most ``top_k``) visible to ``role``, deduplicated.

        With ``RETRIEVAL_CROSS_LANG_PREFETCH`` > 0 and the query's language, a third dense arm
        brings chunks in *other* languages: both main arms favour same-language chunks, so an
        English answer to an Arabic question often never reached the reranker (ISSUES.md I9).
        ``extra_dense`` adds one dense arm per rewrite text (query-rewrite Req 2.2), over the
        chunks of the given language, or all chunks for None.
        """
        flt = visibility_filter(role)
        limit = self.settings.prefetch_limit
        prefetch = [
            qm.Prefetch(query=dense, using=DENSE, limit=limit, filter=flt),
            qm.Prefetch(query=sparse, using=SPARSE, limit=limit, filter=flt),
        ]
        cross = self.settings.cross_lang_prefetch
        if cross and query_language:
            other = qm.Filter(
                must=[flt],
                must_not=[
                    qm.FieldCondition(key="language", match=qm.MatchValue(value=query_language))
                ],
            )
            prefetch.append(qm.Prefetch(query=dense, using=DENSE, limit=cross, filter=other))
        for vector, language in extra_dense:
            arm = flt
            if language:
                same = qm.FieldCondition(key="language", match=qm.MatchValue(value=language))
                arm = qm.Filter(must=[flt, same])
            prefetch.append(qm.Prefetch(query=vector, using=DENSE, limit=limit, filter=arm))
        async with guard("search"):
            response = await self.client.query_points(
                self.alias,
                prefetch=prefetch,
                query=self._fusion(len(prefetch)),
                query_filter=flt,
                limit=self.settings.top_k,
                with_payload=_PAYLOAD,
                with_vectors=False,
            )
        return dedupe([Candidate.from_point(p, rank) for rank, p in enumerate(response.points)])

    async def neighbours(
        self, hits: Sequence[Candidate], role: Role
    ) -> dict[tuple[str, str, int], str]:
        """Text of the chunks at ``index ± 1`` of each hit, same document version, same role."""
        if not hits:
            return {}
        wanted = [
            qm.Filter(
                must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=h.doc_id)),
                    qm.FieldCondition(
                        key="content_hash", match=qm.MatchValue(value=h.content_hash)
                    ),
                    qm.FieldCondition(
                        key="index", match=qm.MatchAny(any=[h.index - 1, h.index + 1])
                    ),
                ]
            )
            for h in hits
        ]
        flt = qm.Filter(must=[visibility_filter(role)], should=wanted)
        async with guard("neighbours"):
            points, _ = await self.client.scroll(
                self.alias,
                scroll_filter=flt,
                limit=2 * len(hits),
                with_payload=["doc_id", "content_hash", "index", "text"],
                with_vectors=False,
            )
        return {
            (p.payload["doc_id"], p.payload["content_hash"], int(p.payload["index"])): p.payload[
                "text"
            ]
            for p in points
            if p.payload
        }
