from mcclub_rag.ingest.chunk import chunk_document
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.rerank import passage
from tests.chunk_helpers import FakeCounter, doc, md, words

SETTINGS = IngestSettings(
    chunk_target_tokens=30, chunk_max_tokens=40, chunk_min_tokens=8, chunk_max_header_tokens=8
)


def test_passage_is_what_the_chunk_was_embedded_with():
    document = doc(
        *md("# Statuts", f"## Adhésion\n\n{words(20)}", f"## Cotisation\n\n{words(25, 'dinar')}"),
        title="Règlement",
    )
    chunks = chunk_document(document, settings=SETTINGS, counter=FakeCounter())
    assert len(chunks) >= 2
    for chunk in chunks:
        assert passage(document.title, chunk.heading_path, chunk.text) == chunk.embed_text


def test_consecutive_repeats_collapse_case_insensitively():
    assert passage("Theming", ["theming", "Overview"], "body") == "Theming > Overview\n\nbody"


def test_blank_entries_are_skipped():
    assert passage("", [" ", "Section"], "body") == "Section\n\nbody"
    assert passage("", [], "body") == "body"


def test_header_off_is_the_text_alone():
    assert passage("Doc", ["Section"], "body", header=False) == "body"
