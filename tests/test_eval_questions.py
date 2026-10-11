"""The eval question set: schema, coverage minimums (Req 10.2) and verbatim evidence."""

import json
from collections import Counter

import pytest

from eval.metrics import fold
from eval.run_eval import document_text
from tests.conftest import REPO

EVAL = REPO / "eval"
CORPUS = json.loads((EVAL / "corpus.json").read_text(encoding="utf-8"))
QUESTIONS = [
    json.loads(line)
    for line in (EVAL / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
DOCS = {d["topic"]: d for d in CORPUS}
KEYS = {"id", "question", "lang", "role", "answerable", "tags", "doc", "evidence", "history"}


def _text(topic: str) -> str:
    raw = (REPO / "data" / "parsed" / DOCS[topic]["file"]).read_text(encoding="utf-8")
    return document_text(raw)  # what the eval ingests


def test_corpus_manifest_points_at_existing_files():
    assert len(DOCS) == len(CORPUS)
    for doc in CORPUS:
        assert (REPO / "data" / "parsed" / doc["file"]).is_file()
        assert doc["visibility"] in ("public", "members")
        assert doc["title"].strip()  # the dumps carry no usable title of their own


@pytest.mark.parametrize("row", QUESTIONS, ids=[r["id"] for r in QUESTIONS])
def test_question_schema(row):
    assert set(row) <= KEYS and {"id", "question", "lang", "role", "answerable", "tags"} <= set(row)
    assert row["lang"] in ("ar", "fr", "en") and row["role"] in ("public", "member")
    assert row["question"].strip() and isinstance(row["answerable"], bool)
    if row["answerable"]:
        assert row["doc"] in DOCS and row["evidence"]
        doc = DOCS[row["doc"]]
        assert row["role"] == "member" or doc["visibility"] == "public"
    for turn in row.get("history", []):
        assert set(turn) == {"user_message", "assistant_message"}


def test_ids_are_unique_and_there_are_enough_questions():
    ids = [r["id"] for r in QUESTIONS]
    assert len(ids) == len(set(ids)) and len(ids) >= 40


def test_coverage_minimums():
    tags = Counter(tag for r in QUESTIONS for tag in r["tags"])
    assert tags["arabic"] >= 10
    assert tags["cross_lingual"] >= 6
    assert tags["unanswerable"] >= 8
    assert tags["follow_up"] >= 4
    assert tags["access"] >= 3
    assert all(r.get("history") for r in QUESTIONS if "follow_up" in r["tags"])
    assert all(
        not r["answerable"] for r in QUESTIONS if {"unanswerable", "access"} & set(r["tags"])
    )


def test_cross_lingual_goes_both_ways():
    langs = {
        (r["lang"], "ar" if r["doc"] in {"aljazeera-gaza-7-october", "bbc-arabic-home"} else "en")
        for r in QUESTIONS
        if "cross_lingual" in r["tags"]
    }
    assert ("ar", "en") in langs and ("en", "ar") in langs and ("fr", "ar") in langs


@pytest.mark.parametrize("row", [r for r in QUESTIONS if r["answerable"]], ids=lambda r: r["id"])
def test_evidence_is_verbatim_in_the_document(row):
    text = fold(_text(row["doc"]))
    for evidence in row["evidence"]:
        assert fold(evidence) in text, evidence


def test_access_questions_target_members_only_documents():
    for row in (r for r in QUESTIONS if "access" in r["tags"]):
        assert row["role"] == "public" and DOCS[row["doc"]]["visibility"] == "members"
