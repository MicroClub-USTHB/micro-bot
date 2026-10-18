"""The adversarial eval corpus (Req 2): manifest and injection markers."""

from eval.e2e.cases import load_corpus
from tests.conftest import REPO

CORPUS_DIR = REPO / "eval" / "e2e" / "corpus"
DOCS = load_corpus(CORPUS_DIR / "manifest.json")
TEXTS = {t: (CORPUS_DIR / d.file).read_text(encoding="utf-8") for t, d in DOCS.items()}


def test_topics_are_marked_and_do_not_clash_with_the_real_corpus():
    real = load_corpus(REPO / "eval" / "corpus.json")
    assert all(topic.startswith("eval-") for topic in DOCS)
    assert not set(DOCS) & set(real)


def test_injection_markers_are_only_in_their_document():
    markers = {t: d.injection_marker for t, d in DOCS.items() if d.injection_marker}
    assert len(markers) >= 2
    for topic, marker in markers.items():
        assert [t for t, text in TEXTS.items() if marker in text] == [topic]


def test_readme_lists_every_file():
    readme = (CORPUS_DIR / "README.md").read_text(encoding="utf-8")
    assert all(doc.file in readme for doc in DOCS.values())
