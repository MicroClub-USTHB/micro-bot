"""Where a run goes wrong (query-rewrite Req 4.4): language groups, failure causes, traits.

Computed on the chat layer, base cases only (no ``~`` robustness variants), so every number is
"one real question, one answer". A failed answerable case gets the first failing step of the
pipeline as its cause: the evidence never reached the LLM (``retrieval_miss``), the LLM
refused although it had it (``refused``), answered without the gold fact (``wrong_answer``),
did not cite the passage holding it (``citation_miss``), was judged unfaithful
(``unfaithful``) or answered in the wrong language (``wrong_language``).
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from eval.e2e.cases import Case
from eval.e2e.score import CaseResult

GROUPS = ("fr", "en", "ar", "darija", "arabizi")
TAGS = ("cross_lingual", "code_switch", "noisy", "followup")
_ANSWER_STEPS = (
    ("contract", "error"),
    ("retrieved", "retrieval_miss"),
    ("behaviour", "refused"),
    ("gold", "wrong_answer"),
    ("evidence", "citation_miss"),
    ("faithfulness", "unfaithful"),
    ("language", "wrong_language"),
)


def group(case: Case) -> str:
    slices = set(case.slices)
    return "arabizi" if "arabizi" in slices else "darija" if "darija" in slices else case.lang


def groups(case: Case) -> list[str]:
    return [group(case), *(t for t in TAGS if t in case.slices)]


def _status(result: CaseResult, name: str) -> str | None:
    return next((c.status for c in result.checks if c.name == name), None)


def cause(result: CaseResult, case: Case) -> str | None:
    """Why the case failed (None when it passed)."""
    if result.passed:
        return None
    failed = {c.name for c in result.failed()}
    if "must_not" in failed or "access" in failed:
        return "leak"
    if case.expect == "answer":
        return next((why for name, why in _ANSWER_STEPS if name in failed), "other")
    if case.expect == "refuse" and "behaviour" in failed:
        return "answered_unanswerable"
    return "other"


def base_chat(results: Sequence[CaseResult]) -> list[CaseResult]:
    return [r for r in results if r.layer == "chat" and "~" not in r.id and r.path != "not_run"]


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def by_group(results: Sequence[CaseResult], cases: Mapping[str, Case]) -> dict[str, dict]:
    """Per group: retrieval hit, answered, correct and cited shares; refusals done right."""
    tally: dict[str, Counter] = defaultdict(Counter)
    for r in base_chat(results):
        case = cases[r.id]
        for g in groups(case):
            t = tally[g]
            if case.expect == "answer":
                t["answerable"] += 1
                t["retrieved_ran"] += _status(r, "retrieved") in ("pass", "fail")
                t["retrieved"] += _status(r, "retrieved") == "pass"
                t["answered"] += bool(r.extra.get("answered"))
                t["correct"] += r.passed
                t["cited"] += _status(r, "evidence") == "pass"
            elif case.expect == "refuse":
                t["unanswerable"] += 1
                t["refused"] += _status(r, "behaviour") == "pass"
    order = [*GROUPS, *TAGS]
    out = {}
    for g in sorted(tally, key=lambda g: order.index(g) if g in order else len(order)):
        t = tally[g]
        out[g] = {
            "answerable": t["answerable"],
            "retrieved": _rate(t["retrieved"], t["retrieved_ran"]),
            "answered": _rate(t["answered"], t["answerable"]),
            "correct": _rate(t["correct"], t["answerable"]),
            "cited": _rate(t["cited"], t["answerable"]),
            "unanswerable": t["unanswerable"],
            "refused": _rate(t["refused"], t["unanswerable"]),
        }
    return out


def attribution(results: Sequence[CaseResult], cases: Mapping[str, Case]) -> dict:
    """Failed cases per cause: counts per language group and the case ids."""
    counts: dict[str, Counter] = defaultdict(Counter)
    ids: dict[str, list[str]] = defaultdict(list)
    for r in base_chat(results):
        why = cause(r, cases[r.id])
        if why:
            counts[why][group(cases[r.id])] += 1
            ids[why].append(r.id)
    return {"counts": {k: dict(v) for k, v in counts.items()}, "ids": dict(ids)}


def traits(results: Sequence[CaseResult], cases: Mapping[str, Case], min_n: int = 3) -> dict:
    """What failed cases share: failure rate per slice (lift over the overall rate), question
    length and history, for answer and refuse cases."""
    rows = [r for r in base_chat(results) if cases[r.id].expect != "any"]
    if not rows:
        return {}
    overall = sum(not r.passed for r in rows) / len(rows)
    per_slice: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for s in cases[r.id].slices:
            per_slice[s].append(not r.passed)
    slices = [
        {
            "slice": s,
            "n": len(f),
            "failed": sum(f),
            "rate": round(sum(f) / len(f), 3),
            "lift": round(sum(f) / len(f) / overall, 2) if overall else None,
        }
        for s, f in per_slice.items()
        if len(f) >= min_n
    ]

    def words(rs: Sequence[CaseResult]) -> float | None:
        values = [len(cases[r.id].question.split()) for r in rs]
        return round(sum(values) / len(values), 1) if values else None

    failed = [r for r in rows if not r.passed]
    passed = [r for r in rows if r.passed]
    return {
        "overall_fail_rate": round(overall, 3),
        "slices": sorted(slices, key=lambda d: (-d["rate"], -d["n"])),
        "words_failed": words(failed),
        "words_passed": words(passed),
        "history_fail_rate": _rate(
            sum(not r.passed for r in rows if cases[r.id].history),
            sum(bool(cases[r.id].history) for r in rows),
        ),
    }
