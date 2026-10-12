"""Retrieval eval: hit@5 / MRR@10 before and after reranking, gate sweep, latency (Req 10).

    uv run python -m eval.run_eval --label baseline
    RETRIEVE_TOP_K=20 RETRIEVAL_RRF_K=2 uv run python -m eval.run_eval --label k20-rrf2
    RERANK_MODEL=BAAI/bge-reranker-v2-m3 uv run python -m eval.run_eval --label bge

Ingests the corpus (``eval/corpus.json``: files in ``--corpus``, topic, title, visibility;
only the document text of ``scripts/parse.py`` dumps) into an in-memory Qdrant with
the real encoders and reranker, runs every question of ``eval/questions.jsonl`` through the
retriever's stages, prints a summary and writes ``eval/results/<UTC>-<label>.json``. Settings
come from the environment exactly as in the app (``RETRIEVAL_*``, ``INGEST_*`` and shared names);
``MODELS_DIR`` defaults to the repo's ``models/``, ``TESSDATA_PREFIX`` to ``$MODELS_DIR/tessdata``.
"""

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval import metrics as m
from mcclub_rag.conversation import Turn
from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.embed import get_dense_encoder, get_sparse_encoder
from mcclub_rag.retrieval.rerank import get_reranker
from mcclub_rag.retrieval.settings import RetrievalSettings

REPO = Path(__file__).resolve().parents[1]
STAGES = ("ms_encode", "ms_search", "ms_rerank", "ms_total")
_DUMP_TEXT = "\n# Text\n"


def document_text(raw: str) -> str:
    """The document part of a ``scripts/parse.py`` dump (metadata and section list dropped).

    The corpus files are those dumps; ingesting them whole indexed the metadata dict and
    section list as content and made "Metadata" every document's title. Other files pass as is.
    """
    if raw.startswith("# Metadata\n") and _DUMP_TEXT in raw:
        return raw.split(_DUMP_TEXT, 1)[1].lstrip("\n")
    return raw


async def ingest_corpus(
    pipeline: IngestPipeline, corpus: Sequence[dict], corpus_dir: Path, work_dir: Path
) -> dict[str, str]:
    """Ingest each manifest entry's document text under its topic and title; topic -> doc_id."""
    work_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = {}
    for doc in corpus:
        path = work_dir / doc["file"]
        raw = (corpus_dir / doc["file"]).read_text(encoding="utf-8")
        path.write_text(document_text(raw), encoding="utf-8")
        result = await pipeline.ingest_file(
            path, path.name, doc["visibility"], title=doc.get("title"), topic=doc["topic"]
        )
        doc_ids[doc["topic"]] = result.doc_id
    return doc_ids


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def evaluate(
    retriever: HybridRetriever, questions: Sequence[dict], doc_ids: dict[str, str], depth: int = 10
) -> list[dict]:
    """One row per question: first-hit ranks (fused / reranked), top score, stage timings."""
    rows = []
    for q in questions:
        history = [Turn(**t) for t in q.get("history", [])]
        query, used = retriever.search_query(q["question"], history)
        start = time.perf_counter()
        dense, sparse = await retriever.encode(query)
        ms_encode = _ms(start)
        step = time.perf_counter()
        candidates = await retriever.search.search(dense, sparse, q["role"])
        ms_search = _ms(step)
        step = time.perf_counter()
        ranked = await retriever.rerank(query, candidates)
        ms_rerank = _ms(step)
        row = {
            "id": q["id"],
            "tags": q["tags"],
            "role": q["role"],
            "answerable": q["answerable"],
            "history_used": used,
            "candidates": len(candidates),
            "top_score": round(ranked[0][1], 4) if ranked else 0.0,
            "visibilities": sorted({c.visibility for c in candidates}),
            "ms_encode": ms_encode,
            "ms_search": ms_search,
            "ms_rerank": ms_rerank,
            "ms_total": _ms(start),
            "rank_fused": None,
            "rank_reranked": None,
        }
        if q["answerable"]:
            expected, evidence = doc_ids[q["doc"]], q["evidence"]
            fused = [(c.doc_id, c.text) for c in candidates[:depth]]
            reranked = [(c.doc_id, c.text) for c, _ in ranked[:depth]]
            row["rank_fused"] = m.first_hit_rank(fused, expected, evidence)
            row["rank_reranked"] = m.first_hit_rank(reranked, expected, evidence)
        rows.append(row)
    return rows


def _quality(rows: Sequence[dict], k: int, depth: int) -> dict[str, float]:
    out: dict[str, float] = {"n": len(rows)}
    for stage in ("fused", "reranked"):
        ranks = [r[f"rank_{stage}"] for r in rows]
        out[f"hit@{k}_{stage}"] = round(m.hit_at(ranks, k), 3)
        out[f"mrr@{depth}_{stage}"] = round(m.mrr_at(ranks, depth), 3)
    return out


def _latency(rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for stage in STAGES:
        values = sorted(r[stage] for r in rows)
        p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))] if values else 0.0
        out[stage] = {"median": statistics.median(values) if values else 0.0, "p95": p95}
    return out


def summarize(
    rows: Sequence[dict], *, k: int = 5, depth: int = 10, max_false_refusal: float = 0.05
) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    tags = sorted({t for r in answerable for t in r["tags"]})
    scores = [r["top_score"] for r in answerable], [r["top_score"] for r in unanswerable]
    sweep = m.sweep(*scores)
    return {
        "overall": _quality(answerable, k, depth),
        "by_tag": {t: _quality([r for r in answerable if t in r["tags"]], k, depth) for t in tags},
        "access_violations": m.access_violations(rows),
        "suggested_gate_threshold": m.suggest_threshold(sweep, max_false_refusal),
        "max_false_refusal": max_false_refusal,
        "latency_ms": _latency(rows),
        "sweep": sweep,
        "gate_scores": {
            "answerable": m.score_distribution(scores[0]),
            "unanswerable": m.score_distribution(scores[1]),
            "overlap": m.score_overlap(*scores),
        },
    }


def write_result(out_dir: Path, label: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}-{label}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")  # one line
    return path


def print_summary(summary: dict, label: str) -> None:
    print(f"\n== {label}")
    for name, q in {"overall": summary["overall"], **summary["by_tag"]}.items():
        cells = "  ".join(f"{key} {value}" for key, value in q.items())
        print(f"{name:>14}: {cells}")
    lat = summary["latency_ms"]
    print("latency ms   : " + "  ".join(f"{s} {v['median']}/{v['p95']}" for s, v in lat.items()))
    print(f"access violations: {summary['access_violations']}")
    if "instruction_like_chunks" in summary:
        print(f"chunks flagged instruction_like: {summary['instruction_like_chunks']}")
    print(
        f"suggested GATE_THRESHOLD: {summary['suggested_gate_threshold']} "
        f"(false refusals <= {summary['max_false_refusal']:.0%})"
    )
    gate = summary["gate_scores"]
    for group in ("answerable", "unanswerable"):
        cells = "  ".join(f"{k} {v}" for k, v in gate[group].items())
        print(f"gate scores {group:>12}: {cells}")
    print(f"gate score overlap (unanswerable >= answerable q1): {gate['overlap']}")
    for row in summary["sweep"][::5]:
        print(
            f"  t={row['threshold']:.2f} refuses {row['refused_unanswerable']:.0%} unanswerable, "
            f"{row['false_refusal']:.0%} answerable"
        )


def _settings_record(retriever: HybridRetriever, pipeline: IngestPipeline) -> dict:
    ingest = pipeline.settings
    return {
        "retrieval": json.loads(retriever.settings.model_dump_json()),
        "chunking": {
            "target": ingest.chunk_target_tokens,
            "max": ingest.chunk_max_tokens,
            "min": ingest.chunk_min_tokens,
            "context_header": ingest.chunk_context_header,
        },
        "dense": pipeline.dense.fingerprint,
        "sparse": pipeline.sparse.fingerprint,
        "reranker": retriever.reranker.fingerprint,
        "index_signature": pipeline.signature[:12],
    }


async def _run(args: argparse.Namespace) -> Path:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qm

    corpus = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ingest = IngestSettings(data_dir=Path(tmp))
        pipeline = IngestPipeline(
            ingest,
            dense=get_dense_encoder(ingest),
            sparse=get_sparse_encoder(ingest),
            client=AsyncQdrantClient(location=":memory:"),
            docstore=DocStore(Path(tmp) / "documents"),
        )
        await pipeline.start()
        started = time.perf_counter()
        doc_ids = await ingest_corpus(pipeline, corpus, args.corpus, Path(tmp) / "corpus")
        print(f"ingested {len(corpus)} documents in {time.perf_counter() - started:.1f}s")
        settings = RetrievalSettings(qdrant_collection=ingest.qdrant_collection)
        retriever = HybridRetriever.from_pipeline(pipeline, settings, get_reranker(settings))
        rows = await evaluate(retriever, questions, doc_ids)
        summary = summarize(rows, max_false_refusal=args.max_false_refusal)
        flagged = await pipeline.client.count(
            pipeline.store.alias,
            count_filter=qm.Filter(
                must=[qm.FieldCondition(key="instruction_like", match=qm.MatchValue(value=True))]
            ),
        )
        summary["instruction_like_chunks"] = flagged.count  # real corpus: target 0 (Req 4.5)
        payload = {
            "label": args.label,
            "settings": _settings_record(retriever, pipeline),
            "corpus_version": await pipeline.corpus_version(),
            "summary": summary,
            "rows": rows,
        }
        await pipeline.close()
    print_summary(summary, args.label)
    return write_result(args.out, args.label, payload)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=REPO / "data" / "parsed")
    parser.add_argument("--manifest", type=Path, default=REPO / "eval" / "corpus.json")
    parser.add_argument("--questions", type=Path, default=REPO / "eval" / "questions.jsonl")
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "results")
    parser.add_argument("--label", default="run")
    parser.add_argument("--max-false-refusal", type=float, default=0.05)
    args = parser.parse_args(argv)
    os.environ.setdefault("MODELS_DIR", str(REPO / "models"))
    os.environ.setdefault("TESSDATA_PREFIX", str(Path(os.environ["MODELS_DIR"]) / "tessdata"))
    print(f"results: {asyncio.run(_run(args))}")


if __name__ == "__main__":
    main()
