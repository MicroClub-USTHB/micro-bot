"""System suites: faults, cache, load (Req 7) and restart.

Probes are checked with ``eval.e2e.contract.expect``. Faults swap parts of the in-process app
state (generator, quota, retrieval) and restore them afterwards.
"""

import asyncio
import resource
import statistics
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence

import httpx

from eval.e2e.cases import Case
from eval.e2e.contract import check, expect, json_body
from eval.e2e.generator import FaultyGenerator
from eval.e2e.score import CaseResult
from mcclub_rag.ingest.errors import IndexUnavailable, RerankError
from mcclub_rag.llm.quota import CircuitBreaker


class _NoQuota:
    def acquire(self) -> bool:
        return False


class _Raising:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def retrieve(self, query, role, history):
        raise self.error


async def faults(driver, probe: Case) -> list[CaseResult]:
    """Req 7.1: each fault on one answerable probe question, state restored after each."""
    state = driver.state
    saved = (state.generator, state.quota, state.retrieval, state.breaker)
    headers = driver.headers(probe.role)

    async def ask() -> httpx.Response:  # a fresh conversation id: no cache hit between faults
        body = {"message": probe.question, "conversation_id": f"eval-{uuid.uuid4()}"}
        return await driver.request("POST", "/api/v1/chat", json=body, headers=headers)

    results = []
    try:
        for kind in ("timeout", "rate_limited", "provider_5xx", "empty"):
            state.generator, state.breaker = FaultyGenerator(kind), CircuitBreaker()
            results.append(expect(f"fault-llm-{kind}", "faults", await ask(), 200, calls=1))
        state.generator, state.breaker = FaultyGenerator("timeout"), CircuitBreaker()
        for _ in range(3):
            await ask()
        results.append(expect("fault-breaker-open", "faults", await ask(), 200, calls=0))
        state.generator, state.breaker, state.quota = saved[0], CircuitBreaker(), _NoQuota()
        results.append(expect("fault-quota-exhausted", "faults", await ask(), 200, calls=0))
        state.quota = saved[1]
        cases = [
            ("index-unavailable", IndexUnavailable("qdrant down"), 503, "index_unavailable"),
            ("rerank-error", RerankError("reranker failed"), 500, "rerank_error"),
            ("unexpected-error", RuntimeError("bug"), 500, "internal"),
        ]
        for name, error, status, code in cases:
            state.retrieval = _Raising(error)
            results.append(expect(f"fault-{name}", "faults", await ask(), status, code))
    finally:
        state.generator, state.quota, state.retrieval, state.breaker = saved
    return results


async def cache(driver, probe: Case, upload: Callable[[], Awaitable[object]]) -> list[CaseResult]:
    """Req 7.3: second identical request is a cache hit; a corpus change or another role misses."""
    body = {"message": probe.question}

    async def ask(headers: dict | None = None) -> httpx.Response:
        return await driver.request("POST", "/api/v1/chat", json=body, headers=headers or {})

    first, second = await ask(), await ask()
    same = json_body(first).get("answer") == json_body(second).get("answer")
    results = [
        expect("cache-first", "cache", first, 200, calls=1),
        expect("cache-hit", "cache", second, 200, calls=0),
        CaseResult("cache-same-answer", "cache", "cache", [check("same", same, "differs")]),
        expect(
            "cache-other-role-misses", "cache", await ask(driver.headers("member")), 200, calls=1
        ),
    ]
    await upload()
    results.append(expect("cache-corpus-change-misses", "cache", await ask(), 200, calls=1))
    return results


async def load(
    driver, cases: Sequence[Case], *, concurrency: int, duration_s: float
) -> tuple[list[CaseResult], dict]:
    """Req 7.4: ``concurrency`` users loop over the cases for ``duration_s`` seconds."""
    latencies, statuses = [], []
    stop = time.monotonic() + duration_s

    async def user(offset: int) -> None:
        i = offset
        while time.monotonic() < stop:
            case = cases[i % len(cases)]
            observation = await driver.chat(case)
            latencies.append(observation.latency_ms)
            statuses.append(observation.status)
            i += concurrency

    started = time.monotonic()
    await asyncio.gather(*(user(n) for n in range(concurrency)))
    elapsed = time.monotonic() - started
    ordered = sorted(latencies) or [0.0]

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    errors = sum(s != 200 for s in statuses)
    stats = {
        "requests": len(statuses),
        "concurrency": concurrency,
        "throughput_rps": round(len(statuses) / elapsed, 2) if elapsed else 0.0,
        "p50_ms": statistics.median(ordered),
        "p95_ms": pick(0.95),
        "p99_ms": pick(0.99),
        "error_rate": round(errors / len(statuses), 4) if statuses else None,
        "rate_limited": statuses.count(429),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    ok = errors == 0
    no_errors = check("no_errors", ok, f"{errors} errors")
    return [CaseResult("load", "load", "load", [no_errors], stats["p95_ms"], stats)], stats


async def restart(
    driver, cases: Sequence[Case], services: Sequence[str] = ("app", "qdrant")
) -> list[CaseResult]:
    """HTTP + compose only: restart each service, wait for health, compare decisions."""

    async def decisions() -> list[bool]:
        return [bool(_json_ok(await driver.chat(c))) for c in cases]

    before, results = await decisions(), []
    for service in services:
        subprocess.run(["docker", "compose", "restart", service], check=True)
        healthy = False
        for _ in range(60):
            try:
                healthy = (await driver.request("GET", "/api/v1/health")).status_code == 200
            except httpx.HTTPError:
                healthy = False
            if healthy:
                break
            await asyncio.sleep(2)
        after = await decisions()
        checks = [
            check("healthy", healthy, "no 200 in 120 s"),
            check("same_decisions", after == before, "decisions changed"),
        ]
        results.append(CaseResult(f"restart-{service}", "restart", "restart", checks))
    return results


def _json_ok(observation) -> bool:
    body = observation.body
    return isinstance(body, dict) and body.get("grounded", False)
