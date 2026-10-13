"""Gate flips and hit@5 under noisy questions, per query-normalization setting (eval-fixes Req 9).

    uv run python -m eval.bench_robust --config base --config fix_case=1,strip_harakat=1

A light stand-in for the e2e robustness suite: the 86 retrieval questions and their variants
(``eval/e2e/perturb.py``: typo, diacritics, noise, filler), one ingest (in-memory Qdrant, real
models), then each configuration. A flip = the gate decision of a variant differs from its base
question's. Retrieval only, no API, no LLM.
"""

import argparse
import asyncio
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from eval import metrics as m
from eval.e2e.perturb import diacritics, filler, noise, typo
from eval.run_eval import REPO, ingest_corpus
from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.embed import get_dense_encoder, get_sparse_encoder
from mcclub_rag.retrieval.settings import RetrievalSettings
from mcclub_rag.text.lang import detect_language

_FIELDS = {"fix_case": "query_fix_case", "strip_harakat": "query_strip_harakat"}


def question_variants(question: str, qid: str) -> dict[str, str]:
    lang = detect_language(question)
    out = {}
    for kind in ("typo", "diacritics", "noise", "filler"):
        rng = random.Random(f"0:{qid}:{kind}")
        if kind == "filler":
            text = filler(question, lang)
        else:
            text = {"typo": typo, "diacritics": diacritics, "noise": noise}[kind](question, rng)
        if text != question:
            out[kind] = text
    return out


def parse_config(raw: str) -> dict[str, bool]:
    if raw == "base":
        return {}
    pairs = (item.split("=") for item in raw.split(","))
    return {_FIELDS[key]: value in ("1", "true") for key, value in pairs}


async def run_config(retriever, questions, doc_ids) -> dict:
    flips: dict[str, list[bool]] = defaultdict(list)
    ranks: dict[str, list[int | None]] = defaultdict(list)
    for q in questions:
        base = await retriever.retrieve(q["question"], q["role"], [])
        texts = {"base": q["question"], **question_variants(q["question"], q["id"])}
        for kind, text in texts.items():
            result = base if kind == "base" else await retriever.retrieve(text, q["role"], [])
            if kind != "base":
                flips[kind].append(result.decision != base.decision)
            if q["answerable"]:
                pairs = [(c.doc_id, c.text) for c in result.chunks]
                ranks[kind].append(m.first_hit_rank(pairs, doc_ids[q["doc"]], q["evidence"]))
    all_flips = [f for values in flips.values() for f in values]
    return {
        "flip_rate": round(sum(all_flips) / len(all_flips), 3),
        "flips": {k: f"{sum(v)}/{len(v)}" for k, v in flips.items()},
        "hit@5": {k: round(m.hit_at(v, 5), 3) for k, v in ranks.items()},
    }


async def main(args) -> None:
    corpus = json.loads((REPO / "eval" / "corpus.json").read_text(encoding="utf-8"))
    questions = [
        json.loads(line)
        for line in (REPO / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and not json.loads(line).get("history")
    ]
    out = {}
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
        for raw in args.config:
            settings = base.model_copy(update=parse_config(raw))
            retriever = HybridRetriever.from_pipeline(pipeline, settings)
            out[raw] = await run_config(retriever, questions, doc_ids)
            print(raw, out[raw], flush=True)
        await pipeline.close()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = REPO / "eval" / "results" / f"{stamp}-robust.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"results: {path}")


if __name__ == "__main__":
    os.environ.setdefault("MODELS_DIR", str(REPO / "models"))
    os.environ.setdefault("TESSDATA_PREFIX", str(Path(os.environ["MODELS_DIR"]) / "tessdata"))
    os.environ.setdefault("GATE_THRESHOLD", "0.40")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=[])
    parsed = parser.parse_args()
    parsed.config = parsed.config or ["base", "fix_case=1", "strip_harakat=1"]
    asyncio.run(main(parsed))
