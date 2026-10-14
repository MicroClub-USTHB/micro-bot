"""Compare two e2e result files (Req 9.4).

    uv run python -m eval.e2e.compare eval/results/e2e/A.json eval/results/e2e/B.json

Prints each metric with its delta, then the cases that went from pass to fail (regressions)
and from fail to pass (fixes). Cases present in only one run are counted, not listed.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _passed(run: dict) -> dict[str, bool]:
    return {r["id"]: r["passed"] for r in run["results"] if r["path"] != "not_run"}


def compare(a: dict, b: dict) -> dict:
    ma, mb = a["summary"]["metrics"], b["summary"]["metrics"]
    metrics = {}
    for name in sorted(set(ma) | set(mb)):
        va, vb = ma.get(name), mb.get(name)
        delta = (
            round(vb - va, 4)
            if isinstance(va, int | float) and isinstance(vb, int | float)
            else None
        )
        metrics[name] = {"a": va, "b": vb, "delta": delta}
    pa, pb = _passed(a), _passed(b)
    both = set(pa) & set(pb)
    return {
        "metrics": metrics,
        "regressions": sorted(i for i in both if pa[i] and not pb[i]),
        "fixes": sorted(i for i in both if not pa[i] and pb[i]),
        "only_a": len(set(pa) - set(pb)),
        "only_b": len(set(pb) - set(pa)),
    }


def render(diff: dict, a: str, b: str) -> str:
    lines = [f"A = {a}", f"B = {b}", "", f"{'metric':24} {'A':>10} {'B':>10} {'delta':>9}"]
    for name, m in diff["metrics"].items():
        delta = "" if m["delta"] is None else f"{m['delta']:+}"
        lines.append(f"{name:24} {m['a']!s:>10} {m['b']!s:>10} {delta:>9}")
    lines += ["", f"regressions ({len(diff['regressions'])}): " + ", ".join(diff["regressions"])]
    lines += [f"fixes ({len(diff['fixes'])}): " + ", ".join(diff["fixes"])]
    lines += [f"only in A: {diff['only_a']}, only in B: {diff['only_b']}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.a, args.b)]
    print(render(compare(*runs), args.a.name, args.b.name))


if __name__ == "__main__":
    main()
