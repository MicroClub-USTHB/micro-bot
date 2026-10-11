import re

import pytest
from structlog.testing import capture_logs

from mcclub_rag.ingest.chunk import _header, chunk_document
from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.settings import IngestSettings
from tests.chunk_helpers import FakeCounter, doc, md, words

COUNTER = FakeCounter()
SETTINGS = IngestSettings(
    chunk_target_tokens=30, chunk_max_tokens=40, chunk_min_tokens=8, chunk_max_header_tokens=8
)
HEADING_LINE = re.compile(r"^#{1,6} .*$", re.M)


def _chunk(document, settings=SETTINGS, counter=COUNTER):
    return chunk_document(document, settings=settings, counter=counter)


class TestFinalize:
    def test_title_equal_to_h1_appears_once(self):
        assert _header("Theming", ("Theming", "Overview"), 8, COUNTER) == "Theming > Overview"

    def test_long_path_keeps_innermost_headings(self):
        header = _header("Club", ("Statuts", "Titre deux", "Article trois"), 5, COUNTER)
        assert header == "Titre deux > Article trois"

    def test_embed_text_is_header_then_text(self):
        (chunk,) = _chunk(doc(*md(f"# Hackathon\n{words(12)}"), title="Events"))
        assert chunk.embed_text == f"Events > Hackathon\n\n{chunk.text}"
        assert "Events" not in chunk.text
        assert chunk.heading_path == ("Hackathon",)

    def test_header_can_be_disabled(self):
        settings = SETTINGS.model_copy(update={"chunk_context_header": False})
        (chunk,) = _chunk(doc(*md(f"# Hackathon\n{words(12)}")), settings=settings)
        assert chunk.embed_text == chunk.text

    def test_superadditive_counter_still_respects_the_cap(self):
        class Superadditive(FakeCounter):
            def count(self, text):
                n = len(text.split())
                return n + n * n // 20

        chunks = _chunk(doc(*md(f"# A\n{words(400)}")), counter=Superadditive())
        assert chunks
        assert all(c.token_count <= 40 for c in chunks)
        assert all(c.token_count == Superadditive().count(c.embed_text) for c in chunks)

    def test_pages_span_merged_sections(self):
        sections = (
            Section(text="## Fees\nPay early.", heading="Fees", page=3, language="en"),
            Section(text="## Refunds\nNone.", heading="Refunds", page=4, language="en"),
        )
        (chunk,) = _chunk(doc(*sections))
        assert (chunk.page_start, chunk.page_end) == (3, 4)

    def test_language_vote_ignores_und_and_falls_back(self):
        sections = (
            Section(text=f"## A\n{words(6)}", heading="A", language="und"),
            Section(text=f"## B\n{words(3)}", heading="B", language="ar"),
        )
        assert _chunk(doc(*sections))[0].language == "ar"
        only_und = (Section(text=f"## A\n{words(12)}", heading="A", language="und"),)
        assert _chunk(doc(*only_und, language="en"))[0].language == "en"
        assert _chunk(doc(*only_und, language="mixed"))[0].language == "und"


DOCUMENTS = {
    "docs_site": md(
        "# Theming",
        f"## Overview\n{words(25)}",
        f"## Add a Theme\n{words(5)}",
        f"### Interactive Mode\n{words(10)}\n\n- step one\n- step two",
        f"### Manual Setup\n```css\n:root {{ --x: 1; }}\n```\n\n{words(50)}",
        f"## Color System\n| Var | Use |\n|---|---|\n| --a | {words(3)} |",
    ),
    "arabic_article": md(
        *(f"## قسم {i}\n" + "\n\n".join([" ".join(["الصمود في غزة."] * 8)] * 5) for i in range(3)),
        language="ar",
    ),
    "slides": md(
        "# VP\nBouras",
        "## Contents",
        *(f"## 0{i}\nPart {i}" for i in range(1, 8)),
        page=2,
    ),
    "xlsx": (
        Section(text="\n".join(f"Nom: N{i} · Rôle: R{i}" for i in range(40)), heading="Membres"),
        Section(text="Date: 12/10 · Lieu: Salle 3", heading="Planning"),
    ),
    "plain": (Section(text=" ".join(f"{words(7)} fin." for _ in range(30))),),
}


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
class TestInvariants:
    def test_contiguous_and_capped(self, name):
        chunks = _chunk(doc(*DOCUMENTS[name]))
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert all(0 < c.token_count <= 40 for c in chunks)

    def test_lossless_coverage(self, name):
        sections = DOCUMENTS[name]
        chunks = _chunk(doc(*sections))
        expected = "".join(HEADING_LINE.sub("", s.text) for s in sections)
        actual = "".join(HEADING_LINE.sub("", c.text) for c in chunks)
        assert "".join(actual.split()) == "".join(expected.split())

    def test_deterministic(self, name):
        assert _chunk(doc(*DOCUMENTS[name])) == _chunk(doc(*DOCUMENTS[name]))


def test_empty_document_gives_no_chunks():
    assert _chunk(doc(Section(text="---"))) == ()


def test_slides_become_one_chunk():
    chunks = _chunk(doc(*DOCUMENTS["slides"]))
    assert len(chunks) == 1
    assert "## 01\nPart 1" in chunks[0].text


def test_logs_one_summary_event_without_text():
    with capture_logs() as logs:
        chunks = _chunk(doc(*DOCUMENTS["docs_site"]))
    events = [e for e in logs if e["event"] == "document_chunked"]
    assert len(events) == 1
    assert events[0]["chunks"] == len(chunks)
    assert {"tokens_min", "tokens_median", "tokens_max", "merged", "hard_splits"} <= set(events[0])
    assert not any("mot1" in str(value) for event in logs for value in event.values())


def test_hard_split_is_logged():
    class CharCounter(FakeCounter):
        def count(self, text):
            return len(text)

        def cut(self, text, max_tokens):
            return text[:max_tokens], text[max_tokens:]

    settings = SETTINGS.model_copy(update={"chunk_context_header": False})
    with capture_logs() as logs:
        chunks = _chunk(doc(Section(text="x" * 100)), settings=settings, counter=CharCounter())
    assert "".join(c.text for c in chunks) == "x" * 100
    assert [e["event"] for e in logs].count("chunk_hard_split") == len(chunks)
