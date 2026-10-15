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

from eval.e2e import insights
from eval.e2e.cases import Case
from eval.e2e.score import CaseResult
from eval.metrics import score_distribution, score_overlap

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

    # Answer rates come from one layer: the chat layer when the run has it. Robustness runs
    # retrieval-layer variants, where an open gate "answers" everything (0.84 false answers).
    chat = [r for r in results if r.layer == "chat"]
    scoped = chat or [r for r in results if r.layer == "retrieval"] or list(results)
    layer_of_retrieval = [r for r in results if r.layer == "retrieval"]

    def expect(value: str, rs: Iterable[CaseResult] = scoped) -> list[CaseResult]:
        return [r for r in rs if case[r.id].expect == value]

    def ran(rs: Iterable[CaseResult], name: str) -> list[bool]:
        """Failed flag of each result where the check ran."""
        return [_failed(r, name) for r in rs if _check_status(r, name) in ("pass", "fail")]

    latencies = [r.latency_ms for r in (chat or results)]
    metrics = {
        "cases": len(results),
        "pass_rate": _rate([r.passed for r in results]),
        "false_answer_rate": _rate(ran(expect("refuse"), "behaviour")),
        "false_refusal_rate": _rate(ran(expect("answer"), "behaviour")),
        "retrieved_rate": _rate([not f for f in ran(chat, "retrieved")]),
        "retrieval_layer_evidence_rate": _rate(
            [not f for f in ran(layer_of_retrieval, "evidence")]
        ),
        "access_leaks": sum(_failed(r, "access") for r in results),
        "must_not_hits": sum(_failed(r, "must_not") for r in results),
        "injection_hits": sum(
            _failed(r, "must_not") for r in results if _INJECTION & set(case[r.id].slices)
        ),
        "contract_failures": sum(_failed(r, "contract") for r in results),
        "language_mismatches": sum(_failed(r, "language") for r in results),
        "evidence_rate": _rate([not f for f in ran(scoped, "evidence")]),
        "gold_rate": _rate([not f for f in ran(scoped, "gold")]),
        "flip_rate": flip_rate(results),
        "instability": instability(results),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
    }
    checks = Counter((c.name, c.status) for r in results for c in r.checks)
    gate = {
        e: [r.extra["top_score"] for r in expect(e) if "top_score" in r.extra]
        for e in ("answer", "refuse")
    }
    return {
        "metrics": metrics,
        "paths": dict(Counter(r.path for r in results)),
        "checks": {f"{name}:{status}": n for (name, status), n in sorted(checks.items())},
        "gate_scores": {
            "answer": score_distribution(gate["answer"]),
            "refuse": score_distribution(gate["refuse"]),
            "overlap": score_overlap(gate["answer"], gate["refuse"]),
        },
        "groups": insights.by_group(results, case),
        "attribution": insights.attribution(results, case),
        "traits": insights.traits(results, case),
        "by_slice": _group(results, lambda r: case[r.id].slices),
        "by_lang": _group(results, lambda r: [case[r.id].lang]),
        "by_role": _group(results, lambda r: [case[r.id].role]),
        "failures": [
            {"id": r.id, "path": r.path, "failed": {c.name: c.reason for c in r.failed()}}
            for r in results
            if not r.passed
        ],
    }


def check_watch(ids: Sequence[str], cases: Mapping[str, Case]) -> None:
    """``--watch`` ids must be real case ids (a typo would silently watch nothing)."""
    unknown = [i for i in ids if i not in cases]
    if unknown:
        raise ValueError(f"unknown --watch case ids: {', '.join(unknown)}")


def watch(results: Sequence[CaseResult], ids: Sequence[str]) -> dict[str, dict[str, str]]:
    """Pass/fail per layer of named cases (fixes Req 10.4, 11.4); variants are not included."""
    out: dict[str, dict[str, str]] = {i: {} for i in ids}
    for r in results:
        if r.id not in out or r.extra.get("repeat", 0) != 0:  # first run only
            continue
        status = "not_run" if r.path == "not_run" else ("pass" if r.passed else "fail")
        out[r.id].setdefault(r.layer, status)
    return out


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
    lines += insights_markdown(summary)
    gate = summary.get("gate_scores", {})
    if any(gate.get(name, {}).get("n") for name in ("answer", "refuse")):
        lines += ["| Gate score | n | min | q1 | median | q3 | max |", "|---" * 7 + "|"]
        for name in ("answer", "refuse"):
            d = gate[name]
            cells = " | ".join(str(d.get(k, "")) for k in ("n", "min", "q1", "median", "q3", "max"))
            lines.append(f"| expect {name} | {cells} |")
        lines += ["", f"Overlap (refuse-expected >= answerable q1): {gate['overlap']}", ""]
    lines += ["| Slice | Pass |", "|---|---|"]
    for name, g in sorted(summary["by_slice"].items(), key=lambda kv: kv[1]["rate"] or 0):
        lines.append(f"| {name} | {g['passed']}/{g['n']} |")
    lines += ["", "| Language | Pass |", "|---|---|"]
    lines += [f"| {k} | {g['passed']}/{g['n']} |" for k, g in summary["by_lang"].items()]
    if summary.get("watched"):
        lines += ["", "## Watched cases", "", "| Case | Chat | Retrieval |", "|---|---|---|"]
        for case_id, layers in summary["watched"].items():
            chat, retrieval = layers.get("chat", "-"), layers.get("retrieval", "-")
            lines.append(f"| {case_id} | {chat} | {retrieval} |")
    if summary["failures"]:
        lines += ["", "<details><summary>All failed results</summary>", ""]
        lines += ["| Case | Path | Failed checks |", "|---|---|---|"]
        for f in summary["failures"]:
            reasons = "; ".join(f"{k}: {v}" for k, v in f["failed"].items()).replace("|", "/")
            lines.append(f"| {f['id']} | {f['path']} | {reasons} |")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return "" if value is None else f"{value:.0%}" if isinstance(value, float) else str(value)


def insights_markdown(summary: Mapping) -> list[str]:
    """Quality per language group, failure causes, and what failed cases share (chat layer)."""
    groups = summary.get("groups") or {}
    if not groups:
        return []
    cols = ("answerable", "retrieved", "answered", "correct", "cited", "unanswerable", "refused")
    lines = ["## Quality by language (chat, base cases)", ""]
    lines += ["| Group | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, g in groups.items():
        lines.append(f"| {name} | " + " | ".join(_cell(g[c]) for c in cols) + " |")
    lines += [
        "",
        "retrieved = evidence in the passages the LLM got; correct = every check passed; "
        "cited = evidence in a cited snippet; refused = unanswerable cases refused.",
        "",
    ]
    counts = summary.get("attribution", {}).get("counts", {})
    if counts:
        names = [g for g in insights.GROUPS if any(g in c for c in counts.values())]
        lines += ["## Why cases failed", ""]
        lines += ["| Cause | total | " + " | ".join(names) + " | cases |"]
        lines += ["|---" * (len(names) + 3) + "|"]
        ids = summary["attribution"]["ids"]
        for why, by in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
            shown = ", ".join(ids[why][:12]) + (" ..." if len(ids[why]) > 12 else "")
            cells = " | ".join(str(by.get(n, "")) for n in names)
            lines.append(f"| {why} | {sum(by.values())} | {cells} | {shown} |")
        lines.append("")
    traits = summary.get("traits") or {}
    if traits.get("slices"):
        lines += ["## What failed cases share", ""]
        lines += [
            f"Overall failure rate {_cell(traits['overall_fail_rate'])}; question length "
            f"{traits['words_failed']} words when failed, {traits['words_passed']} when passed; "
            f"with history {_cell(traits['history_fail_rate'])} failed.",
            "",
            "| Slice | n | failed | rate | lift |",
            "|---|---|---|---|---|",
        ]
        for s in traits["slices"][:12]:
            lines.append(
                f"| {s['slice']} | {s['n']} | {s['failed']} | {_cell(s['rate'])} | {s['lift']} |"
            )
        lines.append("")
    return lines


def load_markdown(stats: Mapping) -> str:
    """Load suite block: throughput, latency, and where the time goes per retrieval stage."""
    lines = [
        "## Load",
        "",
        f"{stats['requests']} requests, {stats['throughput_rps']} req/s, "
        f"p50 {stats['p50_ms']} ms, p95 {stats['p95_ms']} ms",
    ]
    if stats.get("stages"):
        lines += ["", "| Stage | n | p50 ms | p95 ms |", "|---|---|---|---|"]
        for stage, t in stats["stages"].items():
            lines.append(f"| {stage} | {t['n']} | {t['p50']} | {t['p95']} |")
    return "\n".join(lines) + "\n"
