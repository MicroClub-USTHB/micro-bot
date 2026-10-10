"""Retrieval eval: hit@5 / MRR@10 before and after reranking, gate sweep, latency (Req 10).

    uv run python eval/run_eval.py --label baseline
    RETRIEVE_TOP_K=20 RETRIEVAL_RRF_K=2 uv run python eval/run_eval.py --label k20-rrf2
    RERANK_MODEL=BAAI/bge-reranker-v2-m3 uv run python eval/run_eval.py --label bge

Ingests the corpus (``eval/corpus.json``, files in ``--corpus``) into an in-memory Qdrant with
the real encoders and reranker, runs every question of ``eval/questions.jsonl`` through the
retriever's stages, prints a summary and writes ``eval/results/<UTC>-<label>.json``. Settings
come from the environment exactly as in the app (``RETRIEVAL_*``, ``INGEST_*`` and shared names).
"""

import json
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval import metrics as m
from mcclub_rag.conversation import Turn
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.query import build_search_query

REPO = Path(__file__).resolve().parents[1]
STAGES = ("ms_encode", "ms_search", "ms_rerank", "ms_total")


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def evaluate(
    retriever: HybridRetriever, questions: Sequence[dict], doc_ids: dict[str, str], depth: int = 10
) -> list[dict]:
    """One row per question: first-hit ranks (fused / reranked), top score, stage timings."""
    rows = []
    for q in questions:
        history = [Turn(**t) for t in q.get("history", [])]
        query, used = build_search_query(
            q["question"], history, retriever.settings.followup_max_words
        )
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
    sweep = m.sweep([r["top_score"] for r in answerable], [r["top_score"] for r in unanswerable])
    return {
        "overall": _quality(answerable, k, depth),
        "by_tag": {t: _quality([r for r in answerable if t in r["tags"]], k, depth) for t in tags},
        "access_violations": m.access_violations(rows),
        "suggested_gate_threshold": m.suggest_threshold(sweep, max_false_refusal),
        "max_false_refusal": max_false_refusal,
        "latency_ms": _latency(rows),
        "sweep": sweep,
    }


def write_result(out_dir: Path, label: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}-{label}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def print_summary(summary: dict, label: str) -> None:
    print(f"\n== {label}")
    for name, q in {"overall": summary["overall"], **summary["by_tag"]}.items():
        cells = "  ".join(f"{key} {value}" for key, value in q.items())
        print(f"{name:>14}: {cells}")
    lat = summary["latency_ms"]
    print("latency ms   : " + "  ".join(f"{s} {v['median']}/{v['p95']}" for s, v in lat.items()))
    print(f"access violations: {summary['access_violations']}")
    print(
        f"suggested GATE_THRESHOLD: {summary['suggested_gate_threshold']} "
        f"(false refusals <= {summary['max_false_refusal']:.0%})"
    )
    for row in summary["sweep"][::5]:
        print(
            f"  t={row['threshold']:.2f} refuses {row['refused_unanswerable']:.0%} unanswerable, "
            f"{row['false_refusal']:.0%} answerable"
        )
