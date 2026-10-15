"""Embedder x reranker matrix, with and without the query rewrite (query-rewrite Req 4.2).

    uv run python -m eval.bench_models --llm ollama --label m1            # records rewrites
    uv run python -m eval.bench_models --mode replay --label m1-again      # no LLM call

Every e2e case that expects an answer (with evidence) or a refusal runs through the app's own
``HybridRetriever.rank`` for each embedder (one ingest each, in-memory Qdrant) and reranker,
once as typed and once with its rewrite. Rewrites are made once, before the matrix, through the
e2e ``RecordingGenerator`` (recorded or replayed by prompt hash), so every model sees the same
rewrites. Reranker scores are cached by (query, passage) with the time each pair took, so a
pair seen by several embedders is scored once and the rerank time stays exact. Writes
``eval/results/models/<UTC>-<label>.json`` and ``.md`` (``eval/bench_report.py``).
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval.bench_report import markdown
from eval.e2e import build
from eval.e2e.cases import Case, load_cases, load_corpus
from eval.e2e.generator import Budget, OllamaClient, RecordingGenerator, ThreadedClient
from eval.metrics import fold

REPO = Path(__file__).resolve().parents[1]
E2E = REPO / "eval" / "e2e"
MANIFESTS = (REPO / "eval" / "corpus.json", E2E / "corpus" / "manifest.json")
CASE_FILES = (E2E / "cases_retrieval.jsonl", E2E / "cases.jsonl")
EMBEDDERS = {  # model -> MODELS_DIR whose embedder/ holds it
    "BAAI/bge-m3": "models",
    "ibm-granite/granite-embedding-311m-multilingual-r2": "models/ab-granite",
    "ibm-granite/granite-embedding-97m-multilingual-r2": "models/ab-granite97",
    "intfloat/multilingual-e5-small": "models/ab-e5small",
}
RERANKERS = (
    "BAAI/bge-reranker-v2-m3",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
ENGLISH_ONLY = {"cross-encoder/ms-marco-MiniLM-L6-v2"}  # reranks with the English rewrite
DEPTH = 10


class TimedCache:
    """Reranker wrapper: one score per (query, passage), with the time the real call took."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.fingerprint = inner.fingerprint
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}
        self.spent_ms = 0.0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        out = []
        for text in passages:
            key = (query, text)
            if key not in self.cache:
                start = time.perf_counter()
                value = self.inner.score(query, [text])[0]
                self.cache[key] = (value, (time.perf_counter() - start) * 1000)
            out.append(self.cache[key][0])
            self.spent_ms += self.cache[key][1]
        return out


def first_hit(items, case: Case, topics: dict[str, str]) -> int | None:
    """1-based rank of the first chunk of an answer document holding the evidence."""
    evidence = [fold(e) for e in case.evidence]
    for rank, c in enumerate(items, start=1):
        if case.docs and topics.get(c.doc_id) not in case.docs:
            continue
        text = fold(f"{c.title} {c.text}")
        if any(e and e in text for e in evidence):
            return rank
    return None


def select(cases: Sequence[Case], loaded: set[str]) -> list[Case]:
    keep = []
    for c in cases:
        if c.expect == "refuse" or (c.expect == "answer" and c.evidence and set(c.docs) <= loaded):
            keep.append(c)
    return keep


async def make_rewrites(args, cases: Sequence[Case]) -> dict[str, dict]:
    from mcclub_rag.config import get_settings
    from mcclub_rag.llm.client import OpenRouterClient
    from mcclub_rag.llm.rewrite import QueryRewriter

    if args.llm == "ollama":
        inner = ThreadedClient(OllamaClient(args.ollama_model))
        model = f"ollama/{args.ollama_model}"
    else:
        inner, model = ThreadedClient(OpenRouterClient()), get_settings().openrouter_model
    generator = RecordingGenerator(
        inner, E2E / "recordings", args.mode, Budget(args.max_llm_calls), model
    )
    rewriter = QueryRewriter(timeout_s=args.rewrite_timeout, cache_size=0)
    out = {}
    for i, case in enumerate(cases, start=1):
        previous = case.history[-1].user_message if case.history else None
        start = time.perf_counter()
        outcome = await rewriter.rewrite(generator, case.question, previous)
        out[case.id] = {
            "rewrite": outcome.rewrite,
            "error": outcome.error,
            "ms": round((time.perf_counter() - start) * 1000),
            "tokens": [outcome.usage.prompt_tokens, outcome.usage.completion_tokens],
        }
        if i % 25 == 0:
            print(f"rewrites {i}/{len(cases)}", flush=True)
    print(f"rewrites: {generator.calls} provider calls ({model})")
    return out


async def embedder_rows(name, cases, rewrites, args, work: Path, rerankers: dict) -> tuple:
    from qdrant_client import AsyncQdrantClient

    from mcclub_rag.ingest.docstore import DocStore
    from mcclub_rag.ingest.pipeline import IngestPipeline
    from mcclub_rag.ingest.settings import IngestSettings
    from mcclub_rag.retrieval.backend import HybridRetriever
    from mcclub_rag.retrieval.embed import get_dense_encoder, get_sparse_encoder
    from mcclub_rag.retrieval.registry import EMBEDDERS as SPECS
    from mcclub_rag.retrieval.settings import RetrievalSettings

    settings = IngestSettings(
        data_dir=work / "data",
        models_dir=REPO / EMBEDDERS[name],
        embed_model=name,
        embed_dim=SPECS[name].dim,
    )
    pipeline = IngestPipeline(
        settings,
        dense=get_dense_encoder(settings),
        sparse=get_sparse_encoder(settings),
        client=AsyncQdrantClient(location=":memory:"),
        docstore=DocStore(settings.data_dir / "documents"),
    )
    await pipeline.start()
    corpus = load_corpus(*MANIFESTS)
    topics, chunks, start = {}, 0, time.perf_counter()
    for source in build.sources(corpus, work):
        doc = source.doc
        result = await pipeline.ingest_file(
            source.path, source.path.name, doc.visibility, title=doc.title, topic=doc.topic
        )
        topics[result.doc_id] = doc.topic
        chunks += result.chunk_count
    ingest = {"seconds": round(time.perf_counter() - start, 1), "chunks": chunks}
    print(f"{name}: ingested {chunks} chunks in {ingest['seconds']} s", flush=True)
    rows = []
    for rname, reranker in rerankers.items():
        mode = "en" if rname in ENGLISH_ONLY else "chunk_lang"
        rs = RetrievalSettings(
            qdrant_collection=settings.qdrant_collection, rerank_model=rname, rewrite_rerank=mode
        )
        retriever = HybridRetriever.from_pipeline(pipeline, rs, reranker)
        for rewritten in (False, True):
            for case in cases:
                rewrite = rewrites[case.id]["rewrite"] if rewritten else None
                reranker.spent_ms = 0.0
                ranked = await retriever.rank(case.question, case.role, case.history, rewrite)
                reranked = [c for c, _ in ranked.reranked]
                row = {"case": case.id, "embedder": name, "reranker": rname}
                rows.append(
                    row
                    | {
                        "rewrite": rewritten and rewrite is not None,
                        "mode": "rewrite" if rewritten else "typed",
                        "rank": first_hit(reranked[:DEPTH], case, topics),
                        "rank_fused": first_hit(ranked.candidates, case, topics),
                        "top_score": round(ranked.reranked[0][1], 4) if reranked else 0.0,
                        "arms": ranked.arms,
                        "ms_encode": ranked.ms["ms_encode"],
                        "ms_search": ranked.ms["ms_search"],
                        "ms_rerank": round(reranker.spent_ms, 1),
                    }
                )
        print(f"  {rname}: done", flush=True)
    await pipeline.close()
    return rows, ingest


async def run(args) -> Path:
    from mcclub_rag.retrieval.rerank import get_reranker
    from mcclub_rag.retrieval.settings import RetrievalSettings

    os.environ.setdefault("MODELS_DIR", str(REPO / "models"))
    os.environ.setdefault("TESSDATA_PREFIX", str(REPO / "models" / "tessdata"))
    corpus = load_corpus(*MANIFESTS)
    with tempfile.TemporaryDirectory() as tmp:
        loaded = {s.doc.topic for s in build.sources(corpus, Path(tmp))}
    cases = select(load_cases(list(CASE_FILES), corpus), loaded)[: args.limit]
    print(f"{len(cases)} cases", flush=True)
    rewrites = await make_rewrites(args, cases)
    rerankers = {
        name: TimedCache(get_reranker(RetrievalSettings(rerank_model=name)))
        for name in args.rerankers
    }
    rows, ingest = [], {}
    for name in args.embedders:
        with tempfile.TemporaryDirectory() as tmp:
            found, ingest[name] = await embedder_rows(
                name, cases, rewrites, args, Path(tmp), rerankers
            )
            rows += found
    payload = {
        "label": args.label,
        "llm": args.ollama_model if args.llm == "ollama" else "openrouter",
        "cases": {c.id: c.model_dump(include={"lang", "slices", "expect", "role"}) for c in cases},
        "rewrites": {
            k: v | {"rewrite": v["rewrite"].__dict__ if v["rewrite"] else None}
            for k, v in rewrites.items()
        },
        "ingest": ingest,
        "rows": rows,
    }
    out = REPO / "eval" / "results" / "models"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{args.label}"
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    stem.with_suffix(".md").write_text(markdown(payload))
    return stem


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), choices=EMBEDDERS)
    parser.add_argument("--rerankers", nargs="+", default=list(RERANKERS), choices=RERANKERS)
    parser.add_argument("--llm", choices=["ollama", "openrouter"], default="ollama")
    parser.add_argument("--ollama-model", default="gemma4-12b-eval")
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="record")
    parser.add_argument("--max-llm-calls", type=int, default=1000)
    parser.add_argument("--rewrite-timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="matrix")
    stem = asyncio.run(run(parser.parse_args(argv)))
    print(f"results: {stem}.md")


if __name__ == "__main__":
    main()
