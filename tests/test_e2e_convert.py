"""questions.jsonl -> e2e cases (Req 1.3): the generated file is current and loads."""

from collections import Counter

from eval.e2e.cases import load_cases, load_corpus
from eval.e2e.convert import SOURCE, TARGET, convert_row, read_rows, render
from tests.conftest import REPO


def test_generated_cases_are_up_to_date():
    assert TARGET.read_text(encoding="utf-8") == render(read_rows(SOURCE)), (
        "run: uv run python -m eval.e2e.convert"
    )


def test_every_converted_case_loads_and_counts_match():
    rows = read_rows(SOURCE)
    cases = load_cases(TARGET, load_corpus(REPO / "eval" / "corpus.json"))
    assert len(cases) == len(rows)
    assert Counter(c.expect for c in cases) == Counter(
        "answer" if r["answerable"] else "refuse" for r in rows
    )


def test_slices():
    base = {"id": "x", "question": "q", "lang": "ar", "role": "member"}
    plain = convert_row(
        base | {"answerable": True, "doc": "d", "evidence": ["e"], "tags": ["arabic"]}
    )
    assert plain["slices"] == ["lookup"]
    off = convert_row(base | {"id": "un-08", "answerable": False, "tags": ["unanswerable"]})
    assert off["slices"] == ["unanswerable_off_topic"] and "docs" not in off
    access = convert_row(base | {"answerable": False, "doc": "d", "tags": ["access", "arabic"]})
    assert access["slices"] == ["access"] and access["expect"] == "refuse"
    assert access["docs"] == ["d"]
