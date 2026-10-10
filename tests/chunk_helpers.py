"""Shared helpers for the chunking tests: a word-count token counter and a document builder."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from mcclub_rag.ingest.chunk import chunk_document
from mcclub_rag.ingest.models import ParsedDocument, Section
from mcclub_rag.ingest.preprocess import preprocess_url
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.text.sentences import split_words

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeCounter:
    """One token per whitespace-separated word, so sizes in tests are exact and readable."""

    fingerprint = "fake"

    def count(self, text: str) -> int:
        return len(text.split())

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count(t) for t in texts]

    def cut(self, text: str, max_tokens: int) -> tuple[str, str]:
        words = split_words(text)
        head: list[str] = []
        for word in words:
            if word.strip() and self.count("".join(head)) >= max_tokens:
                break
            head.append(word)
        if not head:  # one oversized "word": cut by characters
            return text[:max_tokens], text[max_tokens:]
        cut = len("".join(head))
        return text[:cut], text[cut:]


def words(n: int, word: str = "mot") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


def doc(*sections: Section, title: str = "Doc", language: str = "fr") -> ParsedDocument:
    return ParsedDocument(
        content_hash="ab" * 32,
        title=title,
        source="doc.md",
        source_type="file",
        mime_type="text/markdown",
        visibility="public",
        language=language,
        text="\n\n".join(s.text for s in sections),
        sections=sections,
        ocr_used=False,
        parser="text",
        page_count=None,
        created_at=datetime(2026, 10, 10, tzinfo=UTC),
    )


def md(*texts: str, page: int | None = None, language: str | None = "fr") -> tuple[Section, ...]:
    """Sections from Markdown snippets; the heading is taken from a leading '#' line."""
    out = []
    for text in texts:
        first = text.split("\n", 1)[0]
        heading = first.lstrip("#").strip() if first.startswith("#") else None
        out.append(Section(text=text, heading=heading, page=page, language=language))
    return tuple(out)


MCOLI_GOLDEN = FIXTURES / "chunking" / "mcoli_introduction.json"


async def mcoli_chunk_summary(settings: IngestSettings) -> list[list]:
    """Preprocess the committed mcoli docs page (mocked fetch) and summarize its chunks."""
    html = (FIXTURES / "web" / "mcoli_introduction.html").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=html, headers={"content-type": "text/html"})

    async def resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        document = await preprocess_url(
            "https://mcoli-ui.microclub.info/docs/introduction",
            visibility="public",
            settings=settings,
            client=client,
            resolve=resolve,
        )
    chunks = chunk_document(document, settings=settings)
    return [[list(c.heading_path), c.page_start, c.token_count] for c in chunks]
