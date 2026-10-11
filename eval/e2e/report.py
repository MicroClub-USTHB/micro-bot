"""Summary, scorecard and markdown report of a run (Req 9.1 to 9.3).

Rates count only checks that ran (``pass`` or ``fail``); skips are reported separately. The
scorecard compares flat metrics with ``eval/e2e/thresholds.json`` and decides the exit code:
0 all thresholds hold, 1 at least one fails (2 is for a crashed run, set by the CLI).
"""

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from eval.e2e.cases import Case
from eval.e2e.score import CaseResult

THRESHOLDS = Path(__file__).with_name("thresholds.json")
_INJECTION = {"injection", "indirect_injection", "history_injection", "system_prompt_leak"}


def _rate(flags: Sequence[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def _check_status(result: CaseResult, name: str) -> str | None:
    return next((c.status for c in result.checks if c.name == name), None)


def _failed(result: CaseResult, name: str) -> bool:
    return _check_status(result, name) == "fail"


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] if ordered else 0.0


def _group(results: Iterable[CaseResult], key) -> dict[str, dict]:
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for name in key(result):
            groups[name].append(result)
    return {
        name: {
            "n": len(rs),
            "passed": sum(r.passed for r in rs),
            "rate": _rate([r.passed for r in rs]),
        }
        for name, rs in sorted(groups.items())
    }


def flip_rate(results: Sequence[CaseResult]) -> float | None:
    """Share of base cases whose answered/refused outcome changes in any of its variants."""
    by_base: dict[str, list[bool]] = defaultdict(list)
    base_outcome = {r.id: r.extra.get("answered") for r in results if "~" not in r.id}
    for r in results:
        if "~" in r.id:
            base = r.id.split("~", 1)[0]
            if base in base_outcome:
                by_base[base].append(r.extra.get("answered") != base_outcome[base])
    return _rate([any(flips) for flips in by_base.values()])


def instability(results: Sequence[CaseResult]) -> float | None:
    """Share of repeated cases whose outcome or top documents differ between repeats."""
    runs: dict[str, set] = defaultdict(set)
    for r in results:
        if "repeat" in r.extra:
            runs[r.id].add((r.extra.get("answered"), tuple(r.extra.get("top_topics", ()))))
    return _rate([len(v) > 1 for v in runs.values()])


def summarize(results: Sequence[CaseResult], cases: Mapping[str, Case]) -> dict:
    case = {r.id: cases[r.id] for r in results}

    def expect(value: str) -> list[CaseResult]:
        return [r for r in results if case[r.id].expect == value]

    def ran(rs: Iterable[CaseResult], name: str) -> list[bool]:
        """Failed flag of each result where the check ran."""
        return [_failed(r, name) for r in rs if _check_status(r, name) in ("pass", "fail")]

    latencies = [r.latency_ms for r in results]
    metrics = {
        "cases": len(results),
        "pass_rate": _rate([r.passed for r in results]),
        "false_answer_rate": _rate(ran(expect("refuse"), "behaviour")),
        "false_refusal_rate": _rate(ran(expect("answer"), "behaviour")),
        "access_leaks": sum(_failed(r, "access") for r in results),
        "must_not_hits": sum(_failed(r, "must_not") for r in results),
        "injection_hits": sum(
            _failed(r, "must_not") for r in results if _INJECTION & set(case[r.id].slices)
        ),
        "contract_failures": sum(_failed(r, "contract") for r in results),
        "language_mismatches": sum(_failed(r, "language") for r in results),
        "evidence_rate": _rate([not f for f in ran(results, "evidence")]),
        "gold_rate": _rate([not f for f in ran(results, "gold")]),
        "flip_rate": flip_rate(results),
        "instability": instability(results),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
    }
    checks = Counter((c.name, c.status) for r in results for c in r.checks)
    return {
        "metrics": metrics,
        "paths": dict(Counter(r.path for r in results)),
        "checks": {f"{name}:{status}": n for (name, status), n in sorted(checks.items())},
        "by_slice": _group(results, lambda r: case[r.id].slices),
        "by_lang": _group(results, lambda r: [case[r.id].lang]),
        "by_role": _group(results, lambda r: [case[r.id].role]),
        "failures": [
            {"id": r.id, "path": r.path, "failed": {c.name: c.reason for c in r.failed()}}
            for r in results
            if not r.passed
        ],
    }


def scorecard(metrics: Mapping, thresholds: Sequence[Mapping]) -> list[dict]:
    """One row per threshold: metric, value, op, limit, ok (a missing metric is not ok)."""
    rows = []
    for t in thresholds:
        value = metrics.get(t["metric"])
        if value is None:
            ok = None
        elif t["op"] == "<=":
            ok = value <= t["value"]
        else:
            ok = value >= t["value"]
        rows.append(
            {"metric": t["metric"], "value": value, "op": t["op"], "limit": t["value"], "ok": ok}
        )
    return rows


def exit_code(rows: Sequence[Mapping]) -> int:
    return 1 if any(row["ok"] is False for row in rows) else 0


def load_thresholds(path: Path = THRESHOLDS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown(summary: Mapping, rows: Sequence[Mapping], title: str) -> str:
    lines = [f"# {title}", "", "| Metric | Value | Threshold | OK |", "|---|---|---|---|"]
    for row in rows:
        mark = {True: "yes", False: "**NO**", None: "n/a"}[row["ok"]]
        lines.append(f"| {row['metric']} | {row['value']} | {row['op']} {row['limit']} | {mark} |")
    lines += ["", "Paths: " + ", ".join(f"{k} {v}" for k, v in summary["paths"].items()), ""]
    lines += ["| Slice | Pass |", "|---|---|"]
    for name, g in sorted(summary["by_slice"].items(), key=lambda kv: kv[1]["rate"] or 0):
        lines.append(f"| {name} | {g['passed']}/{g['n']} |")
    lines += ["", "| Language | Pass |", "|---|---|"]
    lines += [f"| {k} | {g['passed']}/{g['n']} |" for k, g in summary["by_lang"].items()]
    if summary["failures"]:
        lines += ["", "## Failed cases", "", "| Case | Path | Failed checks |", "|---|---|---|"]
        for f in summary["failures"]:
            reasons = "; ".join(f"{k}: {v}" for k, v in f["failed"].items()).replace("|", "/")
            lines.append(f"| {f['id']} | {f['path']} | {reasons} |")
    return "\n".join(lines) + "\n"
