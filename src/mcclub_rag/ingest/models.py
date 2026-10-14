"""Output contract of the preprocessing phase, plus internal extraction types."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Visibility = Literal["public", "members"]
SourceType = Literal["file", "web"]
SourceKind = Literal["pdf", "docx", "xlsx", "image", "markdown", "text", "html"]


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    heading: str | None = None
    page: int | None = None  # 1-based; None when the format has no pages
    language: str | None = None  # ISO 639-1 or "und"


class ParsedDocument(BaseModel):
    """One preprocessed document, ready for chunking.

    ``doc_id`` is deliberately absent: ``ingest/pipeline.py`` derives it from the topic.
    """

    model_config = ConfigDict(frozen=True)

    content_hash: str  # sha256 hex of raw bytes (files) / normalized text (web)
    title: str
    source: str  # original filename, or final URL after redirects
    source_type: SourceType
    mime_type: str
    visibility: Visibility
    language: str  # "ar" | "fr" | "en" | "mixed" | "und"
    text: str  # normalized, sections joined by blank lines
    sections: tuple[Section, ...]
    ocr_used: bool
    parser: str  # e.g. "xberg 1.3.6", "trafilatura 2.3.1", "text"
    page_count: int | None
    warnings: tuple[str, ...] = ()
    created_at: datetime  # UTC


class Chunk(BaseModel):
    """One retrieval unit produced by ``ingest/chunk.py`` from a ``ParsedDocument``.

    Chunk ids are deliberately absent: the pipeline derives them from ``doc_id`` and ``index``.
    """

    model_config = ConfigDict(frozen=True)

    index: int  # 0-based, contiguous within the document
    text: str  # faithful content: LLM context and snippets
    embed_text: str  # contextual header + text: dense and BM25 input
    heading_path: tuple[str, ...]  # outermost first; () when the text has no headings
    page_start: int | None
    page_end: int | None
    section_index: int  # index of the first source section in ParsedDocument.sections
    language: str  # "ar" | "fr" | "en" | "und"
    token_count: int  # tokens of embed_text, embedder tokenizer


IngestStatus = Literal["created", "replaced", "duplicate", "updated"]


class IngestResult(BaseModel):
    """What ``ingest_file`` / ``ingest_url`` did with one document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    status: IngestStatus
    title: str
    source: str
    source_type: SourceType
    visibility: Visibility
    language: str
    chunk_count: int
    content_hash: str
    ocr_used: bool  # False for duplicate / updated: nothing was parsed
    warnings: tuple[str, ...] = ()
    duration_ms: int


class DocumentInfo(BaseModel):
    """One complete indexed document, as listed for the admin endpoints."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    source: str
    source_type: SourceType
    visibility: Visibility
    language: str
    chunk_count: int
    content_hash: str
    ingested_at: datetime
    expires_at: datetime | None = None  # after it, the document is not retrieved (I14)


class StoredDocument(BaseModel):
    """``DATA_DIR/documents/<doc_id>.json``: enough to rebuild a document's points."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    doc_id: str
    topic: str
    ingested_at: datetime
    document: ParsedDocument
    expires_at: datetime | None = None


@dataclass(frozen=True)
class RawPage:
    """One page (PDF/image), sheet (XLSX) or the whole body (DOCX) as extracted."""

    number: int
    content: str
    sheet_name: str | None = None
    tables: tuple[tuple[tuple[str, ...], ...], ...] = ()  # tables -> rows -> cells
    ocr: bool = False


@dataclass
class RawExtraction:
    """Extractor output before sectioning and normalization (internal)."""

    kind: SourceKind
    mime_type: str
    parser: str
    markdown: str
    pages: list[RawPage] = field(default_factory=list)
    metadata_title: str | None = None
    ocr_used: bool = False
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)
