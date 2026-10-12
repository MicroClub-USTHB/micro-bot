"""Offline gate calibration from an e2e result file (eval-fixes Req 3.4, 9.3)."""

import json

from eval import calibrate_gate as cg


def _row(case_id, score, layer="retrieval", **extra):
    return {"id": case_id, "layer": layer, "extra": {"top_score": score} | extra}


EXPECT = {"a": "answer", "b": "answer", "c": "answer", "r": "refuse", "x": "any"}
ROWS = [
    _row("a", 0.70),
    _row("a~typo", 0.45),
    _row("b", 0.50),
    _row("b~case", 0.52),
    _row("c", 0.30),
    _row("r", 0.40),
    _row("x", 0.20),
    _row("a", 0.10, repeat=1),  # repeats are ignored
    _row("a", 0.99, layer="chat"),  # chat rows carry no gate score
]


def test_rates_per_threshold():
    rows = cg.points(ROWS, EXPECT)
    assert len(rows) == 7
    at = {r["threshold"]: r for r in cg.sweep(rows, [0.35, 0.48, 0.60])}
    assert at[0.35] == {
        "threshold": 0.35,
        "false_refusal": 1 / 3,
        "false_answer": 1.0,
        "flip_rate": 0.0,
    }
    assert at[0.48]["flip_rate"] == 0.5  # a~typo flips, b~case does not
    assert at[0.60]["false_refusal"] == 2 / 3 and at[0.60]["false_answer"] == 0.0


def test_best_minimises_false_answers_and_flips_within_the_refusal_budget():
    rows = cg.sweep(cg.points(ROWS, EXPECT), [0.35, 0.48, 0.60])
    # 0.35: 1 false answer; 0.48: 1 flip in 2; 0.60: same 0.5 but more refusals -> tie, lower wins
    assert cg.best(rows, max_false_refusal=0.34)["threshold"] == 0.48
    assert cg.best(rows, max_false_refusal=0.7)["threshold"] == 0.48
    assert cg.best(rows, max_false_refusal=0.1) is None


def test_split_filter_keeps_dev_cases(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"results": ROWS}))
    loaded = cg.load_results(path)
    assert len(loaded) == len(ROWS)
    dev = cg.points(loaded, EXPECT, split="dev")
    assert {p.case_id.split("~")[0] for p in dev} <= set(EXPECT)
