"""End-to-end eval CLI (Req 8, 9).

    uv run python -m eval.e2e.run --suite all --label baseline
    uv run python -m eval.e2e.run --suite quality --layer chat --mode record --max-llm-calls 10
    uv run python -m eval.e2e.run --suite quality --mode replay
    uv run python -m eval.e2e.run --suite quality --llm ollama --max-llm-calls 400   # local
    uv run python -m eval.e2e.run --target http --base-url http://localhost:8000 --suite contract

In-process (default): the real models, an in-memory Qdrant, the real corpus (``data/parsed``,
when present) plus the adversarial one, and the app built around a recording generator. Writes
``eval/results/e2e/<UTC>-<label>.json`` and ``.md``; exits 1 when a threshold fails, 2 when
the run crashed.
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval.e2e import build, judge, report, system
from eval.e2e.cases import Case, load_cases, load_corpus, split
from eval.e2e.contract import contract
from eval.e2e.drivers import HttpDriver
from eval.e2e.generator import CitingStub
from eval.e2e.ingest_suite import ingest
from eval.e2e.suites import Context, quality, robustness
from mcclub_rag.llm.quota import CircuitBreaker

REPO = Path(__file__).resolve().parents[2]
E2E = REPO / "eval" / "e2e"
MANIFESTS = (REPO / "eval" / "corpus.json", E2E / "corpus" / "manifest.json")
CASE_FILES = (E2E / "cases_retrieval.jsonl", E2E / "cases.jsonl")
ALL = ("contract", "faults", "quality", "robustness", "cache", "ingest", "load")
INPROCESS_ONLY = {"faults", "cache"}


def select(cases: Sequence[Case], loaded: set[str], args) -> list[Case]:
    chosen = [
        c
        for c in cases
        if set(c.docs) <= loaded or c.expect != "answer"
        if args.split == "all" or split(c.id) == args.split
        if not args.ids or any(c.id.startswith(prefix) for prefix in args.ids)
    ]
    return chosen[: args.limit] if args.limit else chosen


def _sha(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _git_sha() -> str:
    run = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return run.stdout.strip() or "unknown"


async def run(args) -> int:
    os.environ.setdefault("MODELS_DIR", str(REPO / "models"))
    os.environ.setdefault("TESSDATA_PREFIX", str(Path(os.environ["MODELS_DIR"]) / "tessdata"))
    os.environ.setdefault("GATE_THRESHOLD", "0.40")  # calibrated for gte (.env.example)
    suites = ALL if args.suite == ["all"] else args.suite
    corpus = load_corpus(*MANIFESTS)
    all_cases = load_cases(list(CASE_FILES), corpus)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        loaded = build.sources(corpus, work)
        pipeline = generator = None
        if args.target == "http":
            driver, budget, info = HttpDriver(args.base_url), None, {"base_url": args.base_url}
            suites = [s for s in suites if s not in INPROCESS_ONLY]
        else:
            driver, budget, generator, info, pipeline = await build.inprocess(args, work)
        topics = await driver.setup(loaded) if args.target == "inprocess" or args.upload else {}
        ctx = Context(driver, corpus, topics, budget, judge.from_env(args.max_judge_calls))
        cases = select(all_cases, {s.doc.topic for s in loaded}, args)
        by_id = {c.id: c for c in all_cases}
        report.check_watch(getattr(args, "watch", None) or [], by_id)
        probe = by_id.get("adv-09") or next(c for c in cases if c.expect == "answer")
        results, extra = [], {}
        for suite in suites:
            print(f"== {suite}", flush=True)
            if suite == "quality":
                results += await quality(ctx, cases, args.layer)
            elif suite == "robustness":
                base = [c for c in cases if not c.history]
                found, variants = await robustness(ctx, base, repeat=args.repeat)
                by_id |= {v.id: v for v in variants}
                results += found
            elif suite == "contract":
                results += await contract(driver)
            elif suite == "faults":
                results += await system.faults(driver, probe)
            elif suite == "cache":
                # Fresh breaker: earlier suites may have opened it (a spent LLM budget fails
                # every call), and an open breaker would skip the stub (0 calls).
                driver.state.generator, driver.state.breaker = CitingStub(), CircuitBreaker()

                async def bump() -> object:  # any upload changes the corpus version
                    return await driver.upload(loaded[0], "eval-cache-bump")

                results += await system.cache(driver, probe, bump)
                driver.state.generator = generator
            elif suite == "ingest":
                results += await ingest(driver)
            elif suite == "load":
                if args.target == "inprocess":
                    driver.state.generator, driver.state.breaker = CitingStub(), CircuitBreaker()
                found, extra["load"] = await system.load(
                    driver, cases, concurrency=args.concurrency, duration_s=args.duration
                )
                results += found
                if args.target == "inprocess":
                    driver.state.generator = generator
            elif suite == "restart" and args.allow_docker:
                results += await system.restart(driver, cases[:20])
        if args.target == "http" and args.cleanup:
            print(f"deleted {await driver.cleanup()} uploaded documents")
        await driver.close()
        if pipeline is not None:
            await pipeline.close()
    info = info | {"llm_calls_made": getattr(generator, "calls", 0)}
    return write(args, results, by_id, ctx, info, extra)


def write(args, results, by_id, ctx, info, extra) -> int:
    for r in results:  # system probes get a synthetic case so they group by suite
        by_id.setdefault(
            r.id,
            Case.model_construct(
                id=r.id, question="", lang="en", role="public", expect="any", slices=[r.layer]
            ),
        )
    summary = report.summarize(results, by_id)
    if getattr(args, "watch", None):
        summary["watched"] = report.watch(results, args.watch)
    rows = report.scorecard(summary["metrics"], report.load_thresholds())
    payload = {
        "label": args.label,
        "git": _git_sha(),
        "target": args.target,
        "info": info,
        "suites": args.suite,
        "split": args.split,
        "cases_sha": _sha(CASE_FILES),
        "notes": ctx.notes,
        "extra": extra,
        "summary": summary,
        "scorecard": rows,
        "results": [r.to_dict() for r in results],
    }
    out = REPO / "eval" / "results" / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{args.label}"
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    text = report.markdown(summary, rows, f"E2E run {args.label}")
    if "load" in extra:
        text += "\n" + report.load_markdown(extra["load"])
    notes = "".join(f"\n> {note}\n" for note in ctx.notes)
    stem.with_suffix(".md").write_text(text.replace("\n\n", notes + "\n", 1) if notes else text)
    print(text[:3000] + notes + f"\nresults: {stem}.json")
    return report.exit_code(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", nargs="+", default=["all"], choices=["all", *ALL, "restart"])
    parser.add_argument("--target", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--upload", action="store_true", help="http: upload the corpus first")
    parser.add_argument("--cleanup", action="store_true", help="http: delete uploaded documents")
    parser.add_argument("--layer", choices=["chat", "retrieval"], default="chat")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--ids", nargs="*", help="only cases whose id starts with one of these")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--watch", type=lambda v: v.split(","), help="comma-separated case ids to report one by one"
    )
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="record")
    parser.add_argument("--max-llm-calls", type=int, default=10)
    parser.add_argument(
        "--llm",
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="ollama: a local test model (no OpenRouter quota), not the production provider",
    )
    parser.add_argument("--ollama-model", default="gemma4:e4b")
    parser.add_argument("--max-judge-calls", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--allow-docker", action="store_true")
    parser.add_argument("--label", default="run")
    args = parser.parse_args(argv)
    try:
        code = asyncio.run(run(args))
    except Exception as exc:  # report a crashed run distinctly from a failed scorecard
        print(f"run crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
