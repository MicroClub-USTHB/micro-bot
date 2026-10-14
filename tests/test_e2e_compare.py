"""Comparing two e2e runs (Req 9.4)."""

import json

from eval.e2e.compare import compare, main


def _run(metrics, results):
    return {
        "summary": {"metrics": metrics},
        "results": [{"id": i, "passed": p, "path": path} for i, p, path in results],
    }


A = _run(
    {"pass_rate": 0.5, "flip_rate": None},
    [("x", True, "llm"), ("y", False, "llm"), ("z", True, "llm"), ("s", False, "not_run")],
)
B = _run(
    {"pass_rate": 0.75, "flip_rate": 0.1},
    [("x", False, "llm"), ("y", True, "llm"), ("w", True, "llm"), ("s", True, "llm")],
)


def test_deltas_regressions_and_fixes():
    diff = compare(A, B)
    assert diff["metrics"]["pass_rate"] == {"a": 0.5, "b": 0.75, "delta": 0.25}
    assert diff["metrics"]["flip_rate"]["delta"] is None
    assert diff["regressions"] == ["x"] and diff["fixes"] == ["y"]
    assert diff["only_a"] == 1 and diff["only_b"] == 2  # not_run counts as absent


def test_cli(tmp_path, capsys):
    for name, run in (("a.json", A), ("b.json", B)):
        (tmp_path / name).write_text(json.dumps(run))
    main([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    out = capsys.readouterr().out
    assert "regressions (1): x" in out and "+0.25" in out
