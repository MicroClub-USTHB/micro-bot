"""Retrieval throughput under concurrent users: reranker pool and rerank cut (eval-fixes Req 17).

    uv run python -m eval.bench_load --workers 1 2 --rerank-top-k 20 15 10 --users 4

Ingests the eval corpus once (in-memory Qdrant, real models), then:

- quality per rerank cut from **one** rerank pass: each pair is scored on its own (one pair per
  ``session.run``), so reranking the first k fused candidates gives the same scores as a run with
  ``RETRIEVAL_RERANK_TOP_K=k``;
- for each (workers, cut): ``--users`` concurrent loops over the questions for ``--seconds``,
  reporting throughput, p50 / p95 retrieval latency and peak RSS (cumulative: sessions of
  earlier configurations stay loaded).

Retrieval only (no API, no LLM): the rerank stage is where concurrent requests queue.
"""

import argparse
import asyncio
import json
import resource
import statistics
import tempfile
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from eval import metrics as m
from eval.run_eval import REPO, ingest_corpus
from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.embed import get_dense_encoder, get_sparse_encoder
from mcclub_rag.retrieval.settings import RetrievalSettings


def _pick(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] if ordered else 0.0


async def quality(retriever, questions, doc_ids, cuts) -> dict[int, dict[str, float]]:
    ranks: dict[int, list[int | None]] = {k: [] for k in cuts}
    for q in questions:
        if not q["answerable"]:
            continue
        query, _ = retriever.search_query(q["question"], [])
        candidates = await retriever.candidates(query, q["role"])
        scored = dict(
            (c.point_id, s) for c, s in await retriever.rerank(query, candidates[: max(cuts)])
        )
        for k in cuts:
            top = sorted(candidates[:k], key=lambda c: (-scored[c.point_id], c.fused_rank))
            pairs = [(c.doc_id, c.text) for c in top[:10]]
            ranks[k].append(m.first_hit_rank(pairs, doc_ids[q["doc"]], q["evidence"]))
    return {
        k: {"hit@5": round(m.hit_at(r, 5), 3), "mrr@10": round(m.mrr_at(r, 10), 3)}
        for k, r in ranks.items()
    }


async def load(retriever, questions, users: int, seconds: float) -> dict[str, float]:
    latencies: list[float] = []
    stop = time.monotonic() + seconds

    async def user(offset: int) -> None:
        i = offset
        while time.monotonic() < stop:
            q = questions[i % len(questions)]
            start = time.perf_counter()
            await retriever.retrieve(q["question"], q["role"], [])
            latencies.append((time.perf_counter() - start) * 1000)
            i += users

    started = time.monotonic()
    await asyncio.gather(*(user(n) for n in range(users)))
    elapsed = time.monotonic() - started
    return {
        "requests": len(latencies),
        "rps": round(len(latencies) / elapsed, 2),
        "p50_ms": round(statistics.median(latencies)),
        "p95_ms": round(_pick(latencies, 0.95)),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


async def main(args) -> None:
    corpus = json.loads((REPO / "eval" / "corpus.json").read_text(encoding="utf-8"))
    questions = [
        json.loads(line)
        for line in (REPO / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: dict = {"users": args.users, "seconds": args.seconds}
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
        doc_ids = await ingest_corpus(pipeline, corpus, REPO / "data" / "parsed", Path(tmp) / "c")
        base = RetrievalSettings(qdrant_collection=ingest.qdrant_collection)
        retriever = HybridRetriever.from_pipeline(pipeline, base)
        out["quality"] = await quality(retriever, questions, doc_ids, args.rerank_top_k)
        print("quality by rerank cut:", out["quality"], flush=True)
        out["load"] = []
        for workers in args.workers:
            for cut in args.rerank_top_k:
                settings = base.model_copy(update={"rerank_workers": workers, "rerank_top_k": cut})
                row = {"workers": workers, "rerank_top_k": cut}
                row |= await load(
                    HybridRetriever.from_pipeline(pipeline, settings),
                    questions,
                    args.users,
                    args.seconds,
                )
                print(row, flush=True)
                out["load"].append(row)
        await pipeline.close()
    path = REPO / "eval" / "results" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-load.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"results: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rerank-top-k", type=int, nargs="+", default=[20, 15, 10])
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=30.0)
    asyncio.run(main(parser.parse_args()))
