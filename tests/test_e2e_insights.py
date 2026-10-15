"""Language groups, failure causes and traits of a run (query-rewrite Req 4.4)."""

from eval.e2e.cases import Case
from eval.e2e.insights import attribution, by_group, cause, group, traits
from eval.e2e.score import CaseResult, Check

NAMES = ("contract", "retrieved", "behaviour", "gold", "evidence", "must_not", "language")


def _case(id, expect="answer", lang="fr", slices=(), question="q", history=False):
    turns = [{"user_message": "u", "assistant_message": "a"}] if history else []
    return Case(
        id=id,
        question=question,
        lang=lang,
        role="public",
        expect=expect,
        slices=list(slices),
        evidence=["e"] if expect == "answer" else [],
        history=turns,
    )


def _result(id, *fails, answered=True, layer="chat"):
    checks = [Check(n, "fail" if n in fails else "pass") for n in NAMES]
    return CaseResult(id, layer, "llm", checks, 10.0, {"answered": answered})


def test_group_prefers_arabizi_then_darija():
    assert group(_case("a", slices=("arabizi", "darija"))) == "arabizi"
    assert group(_case("a", lang="ar", slices=("darija",))) == "darija"
    assert group(_case("a", lang="en")) == "en"


def test_cause_is_the_first_failing_step():
    case = _case("a")
    assert cause(_result("a"), case) is None
    assert cause(_result("a", "retrieved", "behaviour", "evidence"), case) == "retrieval_miss"
    assert cause(_result("a", "behaviour", "evidence"), case) == "refused"
    assert cause(_result("a", "gold"), case) == "wrong_answer"
    assert cause(_result("a", "evidence"), case) == "citation_miss"
    assert cause(_result("a", "must_not", "gold"), case) == "leak"
    refuse = _case("r", expect="refuse")
    assert cause(_result("r", "behaviour"), refuse) == "answered_unanswerable"


def test_by_group_and_attribution_use_base_chat_results_only():
    cases = {
        "a": _case("a", lang="ar", slices=("darija", "cross_lingual")),
        "b": _case("b", lang="ar", slices=("darija",)),
        "r": _case("r", expect="refuse", lang="ar", slices=("darija",)),
        "a~typo": _case("a~typo", lang="ar", slices=("darija", "noisy")),
    }
    results = [
        _result("a"),
        _result("b", "retrieved", "behaviour", answered=False),
        _result("r"),
        _result("a~typo", "retrieved"),  # variant: ignored
        _result("a", "retrieved", layer="retrieval"),  # other layer: ignored
    ]
    g = by_group(results, cases)
    assert g["darija"] == {
        "answerable": 2,
        "retrieved": 0.5,
        "answered": 0.5,
        "correct": 0.5,
        "cited": 1.0,
        "unanswerable": 1,
        "refused": 1.0,
    }
    assert g["cross_lingual"]["answerable"] == 1
    assert attribution(results, cases) == {
        "counts": {"retrieval_miss": {"darija": 1}},
        "ids": {"retrieval_miss": ["b"]},
    }


def test_traits_rank_slices_by_failure_rate():
    cases = {
        f"c{i}": _case(f"c{i}", slices=("numeric",) if i < 3 else ("lookup",), question="a b c")
        for i in range(6)
    }
    cases["h"] = _case("h", slices=("lookup",), history=True, question="a")
    results = [_result(f"c{i}", "gold" if i < 3 else "x") for i in range(6)]
    results.append(_result("h", "gold"))
    t = traits(results, cases)
    assert t["overall_fail_rate"] == round(4 / 7, 3)
    assert t["slices"][0]["slice"] == "numeric" and t["slices"][0]["rate"] == 1.0
    assert t["words_failed"] == 2.5 and t["words_passed"] == 3.0
    assert t["history_fail_rate"] == 1.0
