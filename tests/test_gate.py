import pytest

from micro_bot.retrieval.gate import decide


def test_no_scores_refuse():
    assert decide([], 0.3) == (0.0, "refuse")
    assert decide([], None) == (0.0, "refuse")


def test_open_gate_answers_whenever_there_is_a_candidate():
    assert decide([0.01], None) == (0.01, "answer")


@pytest.mark.parametrize(
    ("scores", "decision"),
    [([0.2, 0.29], "refuse"), ([0.1, 0.3], "answer"), ([0.9, 0.5], "answer")],
)
def test_top_score_against_threshold(scores, decision):
    assert decide(scores, 0.3) == (max(scores), decision)
