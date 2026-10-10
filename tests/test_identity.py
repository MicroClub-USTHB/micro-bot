import pytest

from mcclub_rag.ingest.errors import InvalidInput
from mcclub_rag.ingest.identity import (
    clean_topic,
    doc_id_for,
    file_topic,
    point_id,
    url_topic,
)


def test_doc_id_is_pinned():
    # Golden values: a change here re-keys every stored document.
    assert doc_id_for("rules") == "4f4eb9e7-e964-51be-8328-f36fb580accb"
    assert (
        doc_id_for("https://mcoli-ui.microclub.info/docs/introduction")
        == "b19de440-3cca-5a89-aea5-19bc696adfd3"
    )


@pytest.mark.parametrize(
    ("filename", "topic"),
    [
        ("rules.pdf", "rules"),
        ("report.v2.pdf", "report.v2"),
        ("notes", "notes"),
        ("dir/sub/Règlement.docx", "Règlement"),
        ("  spaced name .md", "spaced name"),
    ],
)
def test_file_topic(filename, topic):
    assert file_topic(filename) == topic


@pytest.mark.parametrize(
    ("url", "topic"),
    [
        ("https://Example.COM/Docs/Intro", "https://example.com/Docs/Intro"),
        ("HTTPS://example.com/a/", "https://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("https://example.com/a?utm_source=x&id=2#part", "https://example.com/a"),
        ("https://user:secret@example.com/a", "https://example.com/a"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/", "https://example.com/"),
    ],
)
def test_url_topic(url, topic):
    assert url_topic(url) == topic


def test_topic_is_nfc():
    decomposed = "Règlement"
    assert clean_topic(decomposed) == "Règlement"
    assert doc_id_for(clean_topic(decomposed)) == doc_id_for("Règlement")


@pytest.mark.parametrize("topic", ["", "   ", "x" * 201])
def test_bad_topic_rejected(topic):
    with pytest.raises(InvalidInput):
        clean_topic(topic)


def test_200_character_topic_allowed():
    assert clean_topic("x" * 200) == "x" * 200


def test_point_id_is_stable_and_distinct():
    doc_id = doc_id_for("rules")
    first = point_id(doc_id, "a" * 64, 0)
    assert first == point_id(doc_id, "a" * 64, 0)
    assert first != point_id(doc_id, "b" * 64, 0)
    assert first != point_id(doc_id, "a" * 64, 1)
    assert first != point_id(doc_id_for("other"), "a" * 64, 0)
