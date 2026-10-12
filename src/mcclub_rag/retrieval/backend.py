"""``HybridRetriever``: eequaled's ``RetrievalBackend`` (retrieval spec, design "backend").

query -> encode (dense + BM25) -> one hybrid Qdrant query -> dedupe -> rerank (one pair per run,
one rerank at a time) -> top_n -> optional neighbours -> gate. No LLM call. The stages are public
so the eval harness measures exactly what the API runs.
"""

import asyncio
import time
from collections.abc import Sequence

import structlog
from qdrant_client import AsyncQdrantClient

from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.retrieval.embed import DenseEncoder, SparseEncoder
from mcclub_rag.retrieval.gate import decide
from mcclub_rag.retrieval.query import build_search_query, normalize_query
from mcclub_rag.retrieval.rerank import Reranker, get_reranker, passage
from mcclub_rag.retrieval.search import Candidate, HybridSearch, visibility_filter
from mcclub_rag.retrieval.settings import RetrievalSettings, get_retrieval_settings
from mcclub_rag.schemas import Chunk, Retrieval, Role
from mcclub_rag.text.lang import detect_language

log = structlog.get_logger(__name__)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class HybridRetriever:
    def __init__(
        self,
        settings: RetrievalSettings,
        *,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        client: AsyncQdrantClient,
        reranker: Reranker,
        chunk_header: bool = True,  # INGEST_CHUNK_CONTEXT_HEADER: rerank what was embedded
        more_rerankers: Sequence[Reranker] = (),  # RETRIEVAL_RERANK_WORKERS > 1
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.chunk_header = chunk_header
        self.search = HybridSearch(client, settings.qdrant_collection, settings)
        # One rerank per session at a time; a pool of sessions serves concurrent requests.
        self._sessions: asyncio.Queue[Reranker] = asyncio.Queue()
        for session in (reranker, *more_rerankers):
            self._sessions.put_nowait(session)
        if settings.gate_threshold is None:
            log.warning("retrieval_gate_open", reason="GATE_THRESHOLD not calibrated")

    @classmethod
    def from_pipeline(
        cls,
        pipeline: IngestPipeline,
        settings: RetrievalSettings | None = None,
        reranker: Reranker | None = None,
    ) -> "HybridRetriever":
        """Share the pipeline's encoders, Qdrant client and alias."""
        settings = settings or get_retrieval_settings()
        if settings.qdrant_collection != pipeline.settings.qdrant_collection:
            settings = settings.model_copy(
                update={"qdrant_collection": pipeline.settings.qdrant_collection}
            )
        return cls(
            settings,
            dense=pipeline.dense,
            sparse=pipeline.sparse,
            client=pipeline.client,
            reranker=reranker or get_reranker(settings),
            chunk_header=pipeline.settings.chunk_context_header,
            more_rerankers=()
            if reranker
            else [get_reranker(settings, worker=i) for i in range(1, settings.rerank_workers)],
        )

    def search_query(self, query: str, history: Sequence[object]) -> tuple[str, bool]:
        """The text that is encoded and reranked: follow-up merge, then query normalization."""
        s = self.settings
        text, used = build_search_query(query, history, s.followup_max_words, cues=s.followup_cues)
        return normalize_query(text, s.query_casefold_shouting, s.query_strip_harakat), used

    async def encode(self, query: str):
        dense = await asyncio.to_thread(self.dense.embed_query, query)
        sparse = await asyncio.to_thread(self.sparse.embed_query, query)
        return dense, sparse

    def query_language(self, query: str) -> str | None:
        """Language for the cross-language arm, or None when that arm is off."""
        return detect_language(query) if self.settings.cross_lang_prefetch else None

    async def candidates(self, query: str, role: Role) -> list[Candidate]:
        dense, sparse = await self.encode(query)
        return await self.search.search(
            dense, sparse, role, query_language=self.query_language(query)
        )

    async def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> list[tuple[Candidate, float]]:
        """Candidates with their reranker score, best first (tie: better fused rank).

        Chunks flagged ``instruction_like`` are ranked by ``score * instruction_penalty``; the
        returned score stays the model's, so the gate is not moved by the penalty.
        """
        candidates = candidates[: self.settings.rerank_top_k]  # None keeps them all
        if not candidates:
            return []
        texts = [
            passage(c.title, c.heading_path, c.text, header=self.chunk_header) for c in candidates
        ]
        session = await self._sessions.get()
        try:
            scores = await asyncio.to_thread(session.score, query, texts)
        finally:
            self._sessions.put_nowait(session)
        penalty = self.settings.instruction_penalty

        def rank(pair: tuple[Candidate, float]) -> tuple[float, int]:
            candidate, score = pair
            return -score * (penalty if candidate.instruction_like else 1.0), candidate.fused_rank

        return sorted(zip(candidates, scores, strict=True), key=rank)

    async def _expand(self, hits: list[Candidate], role: Role) -> dict[str, str]:
        found = await self.search.neighbours(hits, role)
        texts = {}
        for h in hits:
            before = found.get((h.doc_id, h.content_hash, h.index - 1))
            after = found.get((h.doc_id, h.content_hash, h.index + 1))
            texts[h.point_id] = "\n\n".join(t for t in (before, h.text, after) if t)
        return texts

    async def retrieve(self, query: str, role: Role, history: list[object]) -> Retrieval:
        s = self.settings
        visibility_filter(role)  # unknown role: ValueError before any work
        search_query, used_history = self.search_query(query, history)
        if not search_query:
            return Retrieval(chunks=[], top_score=0.0, decision="refuse")
        started = time.perf_counter()
        dense, sparse = await self.encode(search_query)
        ms_encode = _ms(started)
        step = time.perf_counter()
        candidates = await self.search.search(
            dense, sparse, role, query_language=self.query_language(query)
        )
        ms_search = _ms(step)
        step = time.perf_counter()
        reranked = await self.rerank(search_query, candidates)
        ranked = reranked[: s.top_n]
        ms_rerank = _ms(step)
        hits = [c for c, _ in ranked]
        texts = await self._expand(hits, role) if s.neighbours and hits else {}
        # every reranked score: a penalised chunk pushed out of top_n still counts for the gate
        top_score, decision = decide([score for _, score in reranked], s.gate_threshold)
        chunks = [
            Chunk(
                chunk_id=c.point_id,
                doc_id=c.doc_id,
                title=c.title,
                text=texts.get(c.point_id, c.text),
                score=score,
            )
            for c, score in ranked
        ]
        log.info(
            "retrieval_done",
            role=role,
            candidates=len(candidates),
            returned=len(chunks),
            top_score=round(top_score, 4),
            decision=decision,
            history_used=used_history,
            ms_encode=ms_encode,
            ms_search=ms_search,
            ms_rerank=ms_rerank,
            duration_ms=_ms(started),
        )
        return Retrieval(chunks=chunks, top_score=top_score, decision=decision)
