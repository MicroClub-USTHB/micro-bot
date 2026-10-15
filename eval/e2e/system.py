"""System suites: faults, load (Req 7) and restart.

Probes are checked with ``eval.e2e.contract.expect``. Faults swap parts of the in-process app
state (generator, quota, retrieval) and restore them afterwards.
"""

import asyncio
import statistics
import subprocess
import time
import uuid
from collections.abc import Sequence

import httpx
import structlog

try:
    import resource
except ImportError:  # pragma: no cover - deployed target is Linux; Windows has no rusage
    resource = None  # type: ignore[assignment]

from eval.e2e.cases import Case
from eval.e2e.contract import check, expect
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

    async def retrieve(self, query, role, history, rewrite=None):
        raise self.error


async def faults(driver, probe: Case) -> list[CaseResult]:
    """Req 7.1: each fault on one answerable probe question, state restored after each."""
    state = driver.state
    saved = (state.generator, state.quota, state.retrieval, state.breaker)
    headers = driver.headers(probe.role)

    async def ask() -> httpx.Response:  # a fresh conversation: no stored context between faults
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


STAGES = ("ms_encode", "ms_search", "ms_rerank")


def stage_times(events: Sequence[dict]) -> dict[str, dict[str, float]]:
    """p50 / p95 per retrieval stage from captured ``retrieval_done`` events (fixes Req 17.3)."""
    done = [e for e in events if e.get("event") == "retrieval_done"]
    out = {}
    for stage in STAGES if done else ():
        values = sorted(e[stage] for e in done)
        p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
        out[stage] = {"n": len(values), "p50": statistics.median(values), "p95": p95}
    return out


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

    # In-process only: the app's own log events give the time per stage (none over HTTP).
    # The first request builds the middleware, which reconfigures structlog: do it before the
    # capture, or the capture is replaced.
    await driver.request("GET", "/api/v1/health")
    started = time.monotonic()
    with structlog.testing.capture_logs() as events:
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
        "peak_rss_mb": (
            round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
            if resource is not None
            else None
        ),
        "stages": stage_times(events),
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
