"""Summary, flip rate, instability, scorecard and markdown (Req 9.1 to 9.3)."""

import pytest

from eval.e2e.cases import Case
from eval.e2e.report import (
    check_watch,
    exit_code,
    flip_rate,
    instability,
    load_markdown,
    load_thresholds,
    markdown,
    scorecard,
    summarize,
    watch,
)
from eval.e2e.score import CaseResult, Check


def _case(id, expect="answer", lang="en", slices=("lookup",), role="public"):
    return Case(
        id=id,
        question="q",
        lang=lang,
        role=role,
        expect=expect,
        slices=list(slices),
        evidence=["e"] if expect == "answer" else [],
    )


def _result(id, *fails, answered=True, latency=100.0, **extra):
    names = ["contract", "behaviour", "evidence", "must_not", "access", "language"]
    checks = [
        Check(n, "fail" if n in fails else "pass", "why" if n in fails else "") for n in names
    ]
    return CaseResult(id, "chat", "llm", checks, latency, {"answered": answered} | extra)


CASES = {
    "a": _case("a"),
    "b": _case("b", lang="ar", slices=("cross_lingual",)),
    "r": _case("r", expect="refuse", slices=("injection",)),
    "a~typo": _case("a~typo", slices=("lookup", "noisy")),
}


def test_rates_and_counts():
    results = [
        _result("a"),
        _result("b", "behaviour", "evidence", answered=False, latency=300),
        _result("r", "behaviour", "must_not", "access"),
        _result("a~typo", answered=False),
    ]
    s = summarize(results, CASES)
    m = s["metrics"]
    assert m["cases"] == 4 and m["pass_rate"] == 0.5
    assert m["false_answer_rate"] == 1.0 and m["false_refusal_rate"] == round(1 / 3, 4)
    assert m["access_leaks"] == 1 and m["injection_hits"] == 1 and m["must_not_hits"] == 1
    assert m["evidence_rate"] == 0.75 and m["flip_rate"] == 1.0
    assert s["by_lang"]["ar"] == {"n": 1, "passed": 0, "rate": 0.0}
    assert s["failures"][0] == {
        "id": "b",
        "path": "llm",
        "failed": {"behaviour": "why", "evidence": "why"},
    }


def test_instability_counts_cases_that_change_between_repeats():
    same = [_result("a", repeat=i, top_topics=["x"]) for i in range(3)]
    moved = [_result("b", repeat=0, top_topics=["x"]), _result("b", repeat=1, top_topics=["y"])]
    assert instability(same + moved) == 0.5
    assert flip_rate(same) is None and instability([_result("a")]) is None


def test_scorecard_exit_code_and_missing_metrics():
    rows = scorecard(
        {"access_leaks": 0, "evidence_rate": 0.8},
        [
            {"metric": "access_leaks", "op": "<=", "value": 0},
            {"metric": "evidence_rate", "op": ">=", "value": 0.9},
            {"metric": "flip_rate", "op": "<=", "value": 0.1},
        ],
    )
    assert [r["ok"] for r in rows] == [True, False, None]
    assert exit_code(rows) == 1 and exit_code(rows[:1] + rows[2:]) == 0


def test_shipped_thresholds_name_real_metrics():
    metrics = summarize([_result("a")], CASES)["metrics"]
    assert {t["metric"] for t in load_thresholds()} <= set(metrics)


def test_markdown_lists_failures_and_escapes_pipes():
    results = [_result("a"), _result("b", "evidence")]
    results[1].checks[2] = Check("evidence", "fail", "not in | snippet")
    text = markdown(summarize(results, CASES), scorecard({}, []), "Run")
    assert "| b | llm | evidence: not in / snippet |" in text and text.startswith("# Run")


def test_gate_scores_split_by_expectation_and_shown_in_markdown():
    results = [
        _result("a", top_score=0.8),
        _result("b", top_score=0.4),
        _result("r", top_score=0.6),
        _result("a~typo"),  # chat-level result: no gate score
    ]
    s = summarize(results, CASES)
    gate = s["gate_scores"]
    assert gate["answer"]["n"] == 2 and gate["answer"]["min"] == 0.4
    assert gate["refuse"] == {"n": 1, "min": 0.6, "q1": 0.6, "median": 0.6, "q3": 0.6, "max": 0.6}
    assert gate["overlap"] == 1.0
    text = markdown(s, [], "t")
    assert "| expect refuse | 1 | 0.6 |" in text and "Overlap" in text
    assert "Gate score" not in markdown(summarize([_result("a")], CASES), [], "t")


def test_watched_cases_report_each_layer():
    results = [
        _result("a"),
        CaseResult("a", "retrieval", "answer", [Check("behaviour", "pass", "")], 1.0, {}),
        _result("b", "behaviour"),
    ]
    watched = watch(results, ["a", "b", "r"])
    assert watched == {
        "a": {"chat": "pass", "retrieval": "pass"},
        "b": {"chat": "fail"},
        "r": {},
    }
    text = markdown(summarize(results, CASES) | {"watched": watched}, [], "t")
    assert "## Watched cases" in text and "| b | fail | - |" in text and "| r | - | - |" in text


def test_unknown_watched_id_is_an_error():
    with pytest.raises(ValueError, match="zz"):
        check_watch(["a", "zz"], CASES)


def test_load_markdown_shows_stage_times():
    stats = {"requests": 8, "throughput_rps": 0.7, "p50_ms": 5900, "p95_ms": 7000}
    stats["stages"] = {"ms_rerank": {"n": 8, "p50": 4000, "p95": 5000}}
    text = load_markdown(stats)
    assert "| ms_rerank | 8 | 4000 | 5000 |" in text and "0.7 req/s" in text


def test_answer_rates_use_the_chat_layer_only():
    gate_open = CaseResult("r", "retrieval", "answer", [Check("behaviour", "fail", "gate")])
    results = [_result("a"), _result("r"), gate_open]
    m = summarize(results, CASES)["metrics"]
    assert m["false_answer_rate"] == 0.0  # the open gate's "answer" is not a chat answer
    only_retrieval = summarize([gate_open], CASES)["metrics"]
    assert only_retrieval["false_answer_rate"] == 1.0  # a --layer retrieval run still reports


def test_markdown_has_language_quality_and_causes():
    results = [_result("a"), _result("b", "behaviour", answered=False), _result("r")]
    text = markdown(summarize(results, CASES), [], "t")
    assert "## Quality by language" in text and "## Why cases failed" in text
    assert "| refused | 1 |" in text and "<details>" in text
