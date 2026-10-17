"""``HybridRetriever``: eequaled's ``RetrievalBackend`` (retrieval spec, design "backend").

query -> encode (dense + BM25) -> one hybrid Qdrant query -> dedupe -> rerank (one pair per run,
one rerank at a time) -> top_n -> optional neighbours -> gate. No LLM call here: a query rewrite
made by the chat route (query-rewrite Req 2) only adds dense arms and picks the reranker's query.
The stages are public (``rank``) so the eval harness measures exactly what the API runs.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import structlog
from qdrant_client import AsyncQdrantClient

from micro_bot.ingest.pipeline import IngestPipeline
from micro_bot.llm.rewrite import QueryRewrite
from micro_bot.retrieval.embed import DenseEncoder, SparseEncoder
from micro_bot.retrieval.gate import decide
from micro_bot.retrieval.query import build_search_query, normalize_query
from micro_bot.retrieval.rerank import Reranker, get_reranker, passage
from micro_bot.retrieval.search import Candidate, HybridSearch, visibility_filter
from micro_bot.retrieval.settings import RetrievalSettings, get_retrieval_settings
from micro_bot.schemas import Chunk, Retrieval, Role
from micro_bot.text.lang import detect_language
from micro_bot.text.normalize import fold_for_search

log = structlog.get_logger(__name__)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _key(text: str) -> str:
    return " ".join(fold_for_search(text).split())


@dataclass
class Ranked:
    """Everything one search produced, before top_n and the gate."""

    search_query: str
    used_history: bool
    candidates: list[Candidate]
    reranked: list[tuple[Candidate, float]]
    arms: int = 2
    ms: dict[str, int] = field(default_factory=dict)


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
            log.warning("retrieval_gate_open", reason="GATE_THRESHOLD unset: the LLM refuses")

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
        return normalize_query(text, s.query_fix_case, s.query_strip_harakat), used

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

    def _normalize(self, text: str) -> str:
        s = self.settings
        return normalize_query(text, s.query_fix_case, s.query_strip_harakat)

    def rewrite_arms(
        self, search_query: str, rewrite: QueryRewrite
    ) -> list[tuple[str, str | None]]:
        """(text, chunk language or None) of each extra dense arm; texts equal to the original
        search query or to each other (folded) are dropped."""
        lang_filter = self.settings.rewrite_lang_filter
        seen, arms = {_key(search_query)}, []
        pairs = [(rewrite.query, None)]
        pairs += [(t, lang if lang_filter else None) for lang, t in rewrite.translations.items()]
        for text, lang in pairs:
            text = self._normalize(text)
            if text and _key(text) not in seen:
                seen.add(_key(text))
                arms.append((text, lang))
        return arms

    def rerank_query(self, search_query: str, rewrite: QueryRewrite) -> Callable[[Candidate], str]:
        """The reranker's query for a candidate (``RETRIEVAL_REWRITE_RERANK``)."""
        mode = self.settings.rewrite_rerank
        query = self._normalize(rewrite.query)
        if mode == "original":
            return lambda c: search_query
        if mode == "query":
            return lambda c: query
        texts = {lang: self._normalize(t) for lang, t in rewrite.translations.items()}
        if mode == "en":
            return lambda c: texts.get("en", query)
        return lambda c: texts.get(c.language, query)

    async def rerank(
        self, query: str | Callable[[Candidate], str], candidates: Sequence[Candidate]
    ) -> list[tuple[Candidate, float]]:
        """Candidates with their reranker score, best first (tie: better fused rank).

        ``query`` is one text, or a function giving each candidate its own (a rewrite in the
        chunk's language); candidates sharing a query are scored together, one pair per run.
        Chunks flagged ``instruction_like`` are ranked by ``score * instruction_penalty``; the
        returned score stays the model's, so the gate is not moved by the penalty.
        """
        candidates = candidates[: self.settings.rerank_top_k]  # None keeps them all
        if not candidates:
            return []
        texts = [
            passage(c.title, c.heading_path, c.text, header=self.chunk_header) for c in candidates
        ]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            groups[query(c) if callable(query) else query].append(i)

        def score_all(session: Reranker) -> list[float]:
            scores = [0.0] * len(candidates)
            for text, indexes in groups.items():
                found = session.score(text, [texts[i] for i in indexes])
                for i, value in zip(indexes, found, strict=True):
                    scores[i] = value
            return scores

        session = await self._sessions.get()
        try:
            scores = await asyncio.to_thread(score_all, session)
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

    async def rank(
        self, query: str, role: Role, history: Sequence[object], rewrite: QueryRewrite | None
    ) -> Ranked | None:
        """Encode, search and rerank; None for an empty query (nothing to search)."""
        search_query, used_history = self.search_query(query, history)
        if not search_query:
            return None
        started = time.perf_counter()
        arms = self.rewrite_arms(search_query, rewrite) if rewrite else []
        bm25_text = "\n".join([search_query, *(text for text, _ in arms)])
        dense = await asyncio.to_thread(self.dense.embed_query, search_query)
        sparse = await asyncio.to_thread(self.sparse.embed_query, bm25_text)
        extra = [(await asyncio.to_thread(self.dense.embed_query, t), lang) for t, lang in arms]
        ms_encode = _ms(started)
        step = time.perf_counter()
        # A rewrite brings its own translations: the cross-language arm is for raw queries.
        language = None if rewrite else self.query_language(query)
        candidates = await self.search.search(
            dense, sparse, role, query_language=language, extra_dense=extra
        )
        ms_search = _ms(step)
        step = time.perf_counter()
        rerank_query = self.rerank_query(search_query, rewrite) if rewrite else search_query
        reranked = await self.rerank(rerank_query, candidates)
        ms = {"ms_encode": ms_encode, "ms_search": ms_search, "ms_rerank": _ms(step)}
        return Ranked(search_query, used_history, candidates, reranked, 2 + len(extra), ms)

    async def retrieve(
        self,
        query: str,
        role: Role,
        history: list[object],
        *,
        rewrite: QueryRewrite | None = None,
    ) -> Retrieval:
        s = self.settings
        visibility_filter(role)  # unknown role: ValueError before any work
        started = time.perf_counter()
        result = await self.rank(query, role, history, rewrite)
        if result is None:
            return Retrieval(chunks=[], top_score=0.0, decision="refuse")
        candidates, reranked = result.candidates, result.reranked
        ranked = reranked[: s.top_n]
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
            history_used=result.used_history,
            rewritten=rewrite is not None,
            arms=result.arms,
            **result.ms,
            duration_ms=_ms(started),
        )
        return Retrieval(chunks=chunks, top_score=top_score, decision=decision)
