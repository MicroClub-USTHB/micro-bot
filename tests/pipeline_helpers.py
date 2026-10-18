"""Shared helpers for the pipeline tests: fake dense encoder, fake preprocessing, factory.

Everything runs without model files: 8-dim hash vectors, the real BM25 encoder on tiny
stopword files, the word-count token counter and Qdrant's in-memory mode.
"""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from qdrant_client import AsyncQdrantClient

from micro_bot.ingest.docstore import DocStore
from micro_bot.ingest.errors import EmptyDocument, UnsupportedFileType
from micro_bot.ingest.models import ParsedDocument, Section
from micro_bot.ingest.pipeline import IngestPipeline
from micro_bot.ingest.settings import IngestSettings
from micro_bot.retrieval.embed import Bm25SparseEncoder
from tests.chunk_helpers import FakeCounter

STOPWORDS = {"arabic.txt": "في\n", "french.txt": "le\nla\n", "english.txt": "the\n"}


class FakeDenseEncoder:
    dim = 8

    def __init__(self, fingerprint: str = "fake-dense") -> None:
        self.fingerprint = fingerprint
        self.calls = 0
        self.fail_after: int | None = None  # raise on the call after this many texts

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values = [b + 1.0 for b in digest[: self.dim]]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_after is not None:
            from micro_bot.ingest.errors import EmbeddingError

            raise EmbeddingError("embedding model failed: test")
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def _sections(text: str) -> tuple[Section, ...]:
    blocks = [b.strip() for b in re.split(r"(?m)^(?=# )", text) if b.strip()]
    return tuple(
        Section(
            text=b,
            heading=b.splitlines()[0].lstrip("# ") if b.startswith("#") else None,
            language="fr",
        )
        for b in blocks
    )


def _doc(text: str, *, source, source_type, title, content_hash) -> ParsedDocument:
    sections = _sections(text)
    if not sections:
        raise EmptyDocument("no readable text found")
    return ParsedDocument(
        content_hash=content_hash,
        title=title or (sections[0].heading or "Untitled"),
        source=source,
        source_type=source_type,
        mime_type="text/markdown",
        language="fr",
        text=text,
        sections=sections,
        ocr_used=False,
        parser="fake",
        page_count=None,
        created_at=datetime(2026, 10, 10, tzinfo=UTC),
    )


class FakePreprocess:
    """Markdown-only stand-in for preprocess_file / preprocess_url that records its calls."""

    def __init__(self) -> None:
        self.file_calls: list[str] = []
        self.url_calls: list[str] = []
        self.pages: dict[str, tuple[str, str]] = {}  # url -> (final url, markdown)

    async def file(self, data, filename, *, title=None, settings=None):
        self.file_calls.append(filename)
        if not filename.endswith((".md", ".txt")):
            raise UnsupportedFileType("unsupported file type: application/octet-stream")
        return _doc(
            data.decode(),
            source=filename,
            source_type="file",
            title=title,
            content_hash=hashlib.sha256(data).hexdigest(),
        )

    async def url(self, url, *, title=None, settings=None):
        self.url_calls.append(url)
        final, text = self.pages[urlsplit(url)._replace(query="", fragment="").geturl()]
        return _doc(
            text,
            source=final,
            source_type="web",
            title=title,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )


def write_stopwords(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in STOPWORDS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in STOPWORDS}


def make_pipeline(
    tmp_path: Path,
    *,
    client: AsyncQdrantClient | None = None,
    dense: FakeDenseEncoder | None = None,
    preprocess: FakePreprocess | None = None,
    **settings,
) -> IngestPipeline:
    fields = {
        "chunk_target_tokens": 30,
        "chunk_max_tokens": 60,
        "chunk_min_tokens": 5,
        "chunk_max_header_tokens": 10,
        "data_dir": tmp_path / "data",
    }
    s = IngestSettings(**(fields | settings))
    hashes = write_stopwords(tmp_path / "bm25")
    preprocess = preprocess or FakePreprocess()
    return IngestPipeline(
        s,
        dense=dense or FakeDenseEncoder(),
        sparse=Bm25SparseEncoder(
            tmp_path / "bm25", k=s.bm25_k, b=s.bm25_b, avg_len=s.bm25_avg_len, expected=hashes
        ),
        client=client or AsyncQdrantClient(location=":memory:"),
        docstore=DocStore(s.data_dir / "documents"),
        counter=FakeCounter(),
        preprocess_file=preprocess.file,
        preprocess_url=preprocess.url,
    )
