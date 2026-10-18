"""E2E eval cases: loading, validation, dev/test split, must_not expansion (Req 1, 2.2, 2.3)."""

import json
from collections import Counter

import pytest

from eval.e2e.cases import EvalDataError, load_cases, load_corpus, split

MANIFEST = [
    {"file": "a.md", "topic": "theming", "title": "Theming"},
    {"file": "b.md", "topic": "board", "title": "Board"},
    {
        "file": "c.md",
        "topic": "notice",
        "title": "Notice",
        "injection_marker": "PWNED-1234",
    },
]
ANSWER = {
    "id": "en-01",
    "question": "How many themes?",
    "lang": "en",
    "expect": "answer",
    "docs": ["theming"],
    "evidence": ["5 signature themes"],
    "slices": ["lookup"],
}


def _write(tmp_path, name, rows):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    return load_corpus(_write(tmp_path, "manifest.json", MANIFEST))


def _load(tmp_path, corpus, rows):
    return load_cases(_write(tmp_path, "cases.jsonl", rows), corpus)


def test_valid_case_loads(tmp_path, corpus):
    (case,) = _load(tmp_path, corpus, [ANSWER])
    assert case.expect == "answer" and case.docs == ["theming"] and case.gold == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"answerable": True}, "answerable"),  # unknown key
        ({"docs": ["theming", "nope"]}, r"unknown docs \['nope'\]"),
        ({"evidence": [], "gold": []}, "needs gold or evidence"),
        ({"slices": ["lookups"]}, "lookups"),
        ({"lang": "de"}, "lang"),
    ],
)
def test_invalid_case_names_the_id(tmp_path, corpus, change, message):
    with pytest.raises(EvalDataError, match=message) as err:
        _load(tmp_path, corpus, [ANSWER | change])
    assert "en-01" in str(err.value)


def test_duplicate_id(tmp_path, corpus):
    with pytest.raises(EvalDataError, match="duplicate id 'en-01'"):
        _load(tmp_path, corpus, [ANSWER, ANSWER])


def test_refuse_needs_no_evidence(tmp_path, corpus):
    row = {k: v for k, v in ANSWER.items() if k not in ("docs", "evidence")}
    (case,) = _load(tmp_path, corpus, [row | {"expect": "refuse"}])
    assert case.expect == "refuse"


def test_every_case_gets_the_injection_markers(tmp_path, corpus):
    other = ANSWER | {"id": "en-02", "must_not": ["secret"]}
    plain, other = _load(tmp_path, corpus, [ANSWER, other])
    assert plain.must_not == ["PWNED-1234"]
    assert other.must_not == ["secret", "PWNED-1234"]


def test_duplicate_topic_across_manifests(tmp_path):
    first = _write(tmp_path, "one.json", MANIFEST[:1])
    second = _write(tmp_path, "two.json", MANIFEST[:1])
    with pytest.raises(EvalDataError, match="duplicate topic 'theming'"):
        load_corpus(first, second)


def test_split_is_stable_and_about_80_20():
    ids = [f"q-{i}" for i in range(1000)]
    counts = Counter(split(i) for i in ids)
    assert 150 <= counts["test"] <= 250
    assert [split(i) for i in ids] == [split(i) for i in ids]


def test_variants_follow_their_base_case():
    assert all(split(f"q-{i}~typo") == split(f"q-{i}") for i in range(200))


def test_ids_are_unique_across_files(tmp_path, corpus):
    first = _write(tmp_path, "one.jsonl", [ANSWER])
    second = _write(tmp_path, "two.jsonl", [ANSWER])
    with pytest.raises(EvalDataError, match=r"two.jsonl:1 'en-01': duplicate id"):
        load_cases([first, second], corpus)


def test_any_needs_no_evidence_and_keeps_must_not(tmp_path, corpus):
    row = {k: v for k, v in ANSWER.items() if k not in ("docs", "evidence")}
    (case,) = _load(tmp_path, corpus, [row | {"expect": "any", "must_not": ["vote for"]}])
    assert case.expect == "any" and case.must_not[0] == "vote for"
