"""The hand-written e2e cases (Req 1.4, 11): coverage, and labels true to the documents.

Coverage runs everywhere. Evidence and gold are checked against the document text: always for
the committed adversarial corpus, and for the real corpus only when ``data/`` is present (it is
gitignored). Scans are checked against their source text (``data/sources/scans/<name>.txt``),
not the OCR output, so an OCR miss shows up in the eval, not here.
"""

from collections import Counter

import pytest

from eval.e2e.cases import SLICES, load_cases, load_corpus
from eval.metrics import fold
from eval.run_eval import document_text
from tests.conftest import REPO

E2E = REPO / "eval" / "e2e"
ADVERSARIAL = E2E / "corpus"
PARSED = REPO / "data" / "parsed"
SCANS = REPO / "data" / "sources" / "scans"
CORPUS = load_corpus(REPO / "eval" / "corpus.json", ADVERSARIAL / "manifest.json")
CASES = load_cases(E2E / "cases.jsonl", CORPUS)
# ids must stay unique with the generated retrieval cases, which the harness loads too
ALL = load_cases([E2E / "cases_retrieval.jsonl", E2E / "cases.jsonl"], CORPUS)
LABELLED = [c for c in CASES if c.evidence]
QUESTIONS = (REPO / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()


def _texts(topic: str) -> list[str] | None:
    """Folded texts that hold the truth for a topic; None when the file is not here."""
    doc = CORPUS[topic]
    if (ADVERSARIAL / doc.file).is_file():
        return [fold((ADVERSARIAL / doc.file).read_text(encoding="utf-8"))]
    parsed = PARSED / doc.file
    if not parsed.is_file():
        return None
    texts = [fold(document_text(parsed.read_text(encoding="utf-8")))]
    truth = SCANS / doc.file.replace(".md", ".txt")
    if truth.is_file():
        texts.append(fold(truth.read_text(encoding="utf-8")))
    return texts


def test_size_and_balance():
    assert len(CASES) >= 200 and len(ALL) == len(CASES) + sum(1 for q in QUESTIONS if q.strip())
    langs = Counter(c.lang for c in CASES)
    assert min(langs.values()) >= 30, langs
    assert sum(c.expect == "refuse" for c in CASES) >= 40


def test_every_slice_has_at_least_three_cases():
    counts = Counter(s for c in CASES for s in c.slices)
    assert {s: counts[s] for s in SLICES if counts[s] < 3} == {}


def test_every_case_has_a_slice_and_answers_name_their_documents():
    assert all(c.slices for c in CASES)
    assert all(c.docs for c in CASES if c.expect == "answer")


def test_access_cases_ask_as_public_about_members_documents():
    for case in (c for c in CASES if "access" in c.slices):
        assert case.role == "public" and case.expect == "refuse", case.id
        assert all(CORPUS[t].visibility == "members" for t in case.docs), case.id


def test_public_answers_only_need_public_documents():
    for case in (c for c in CASES if c.role == "public" and c.expect == "answer"):
        assert any(CORPUS[t].visibility == "public" for t in case.docs), case.id


@pytest.mark.parametrize("case", LABELLED, ids=[c.id for c in LABELLED])
def test_labels_are_in_the_documents(case):
    found = {t: _texts(t) for t in case.docs}
    texts = [text for value in found.values() if value for text in value]
    if not texts:
        pytest.skip("documents not present (data/ is gitignored)")
    for evidence in case.evidence:
        assert any(fold(evidence) in text for text in texts), evidence
    if case.gold:
        assert any(fold(g) in text for g in case.gold for text in texts), case.gold
