"""Calibrate GATE_THRESHOLD offline from an e2e result file (eval-fixes Req 3.4, 9.3).

    uv run python -m eval.calibrate_gate eval/results/e2e/<run>.json --split dev

The run must include retrieval-layer results with the robustness variants (``--suite quality
robustness``; chat answers are not needed). Each threshold is scored on false refusals
(answerable cases refused), false answers (refuse-expected cases answered) and the flip rate
(base cases whose decision changes in any noisy variant). The suggested value has false
refusals within the budget and the fewest false answers + flips; ties go to the lower threshold.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.e2e.cases import load_cases, load_corpus
from eval.e2e.cases import split as case_split

REPO = Path(__file__).resolve().parents[1]
E2E = REPO / "eval" / "e2e"


@dataclass(frozen=True)
class Point:
    case_id: str
    expect: str  # answer / refuse / any
    score: float


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def points(
    results: Iterable[Mapping], expect: Mapping[str, str], split: str = "all"
) -> list[Point]:
    """First-run retrieval-layer scores; a variant ``<id>~<kind>`` takes its base's expect."""
    seen, out = set(), []
    for r in results:
        extra = r.get("extra", {})
        if r["layer"] != "retrieval" or "top_score" not in extra or extra.get("repeat", 0):
            continue
        base = r["id"].split("~", 1)[0]
        if r["id"] in seen or base not in expect:
            continue
        if split != "all" and case_split(r["id"]) != split:
            continue
        seen.add(r["id"])
        out.append(Point(r["id"], expect[base], extra["top_score"]))
    return out


def _share(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def sweep(pts: Sequence[Point], thresholds: Iterable[float]) -> list[dict]:
    bases = {p.case_id: p for p in pts if "~" not in p.case_id}
    variants: dict[str, list[Point]] = defaultdict(list)
    for p in pts:
        if "~" in p.case_id and p.case_id.split("~", 1)[0] in bases:
            variants[p.case_id.split("~", 1)[0]].append(p)
    rows = []
    for t in thresholds:
        answer = [p.score < t for p in bases.values() if p.expect == "answer"]
        refuse = [p.score >= t for p in bases.values() if p.expect == "refuse"]
        flips = [
            any((v.score >= t) != (bases[b].score >= t) for v in vs) for b, vs in variants.items()
        ]
        rows.append(
            {
                "threshold": round(t, 4),
                "false_refusal": _share(answer),
                "false_answer": _share(refuse),
                "flip_rate": _share(flips),
            }
        )
    return rows


def best(rows: Sequence[dict], max_false_refusal: float) -> dict | None:
    ok = [r for r in rows if r["false_refusal"] <= max_false_refusal]
    return min(ok, key=lambda r: (r["false_answer"] + r["flip_rate"], r["threshold"]), default=None)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path, help="eval/results/e2e/<run>.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--max-false-refusal", type=float, default=0.10)
    parser.add_argument("--low", type=float, default=0.20)
    parser.add_argument("--high", type=float, default=0.60)
    args = parser.parse_args(argv)
    corpus = load_corpus(REPO / "eval" / "corpus.json", E2E / "corpus" / "manifest.json")
    cases = load_cases([E2E / "cases_retrieval.jsonl", E2E / "cases.jsonl"], corpus)
    pts = points(load_results(args.result), {c.id: c.expect for c in cases}, args.split)
    steps = round((args.high - args.low) / 0.01)
    rows = sweep(pts, [args.low + i * 0.01 for i in range(steps + 1)])
    print(f"{len(pts)} scores ({args.split}), false refusals <= {args.max_false_refusal:.0%}")
    print("threshold  false_refusal  false_answer  flip_rate")
    for r in rows:
        print(
            f"{r['threshold']:9.2f}  {r['false_refusal']:13.1%}  {r['false_answer']:12.1%}"
            f"  {r['flip_rate']:9.1%}"
        )
    choice = best(rows, args.max_false_refusal)
    print(f"suggested GATE_THRESHOLD: {choice['threshold'] if choice else None}")


if __name__ == "__main__":
    main()
