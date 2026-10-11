"""Entry points of the preprocessing phase: bytes or URL -> ParsedDocument.

Steps: validate -> detect type -> extract (xberg / text decode / trafilatura) -> sections ->
normalize -> language -> ParsedDocument, with one structlog event per document. CPU-bound
pure-Python steps run in a worker thread; xberg is natively async. A per-event-loop
semaphore bounds how many documents are processed at once.
"""

import asyncio
import hashlib
import time
import weakref
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import httpx
import structlog

from mcclub_rag.ingest.errors import EmptyDocument, FileTooLarge, IngestError, InvalidInput
from mcclub_rag.ingest.models import ParsedDocument, RawExtraction, Section, SourceType
from mcclub_rag.ingest.ocr import verify_tessdata
from mcclub_rag.ingest.parse import extract_file
from mcclub_rag.ingest.sections import build_sections
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings
from mcclub_rag.ingest.sniff import detect_type
from mcclub_rag.ingest.url_guard import Resolver, default_resolver
from mcclub_rag.ingest.web import extract_html, fetch
from mcclub_rag.text.lang import assign_languages
from mcclub_rag.text.normalize import normalize_text

log = structlog.get_logger(__name__)

_VISIBILITIES = ("public", "members")
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_TYPE_EXT = {"text/plain": ".txt", "text/markdown": ".md"}
_MAX_TITLE = 200

_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)
_verified_tessdata: set[tuple[Path, tuple[str, ...]]] = set()


def _semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if (semaphore := _semaphores.get(loop)) is None:
        semaphore = _semaphores[loop] = asyncio.Semaphore(limit)
    return semaphore


def _verify_tessdata_once(settings: IngestSettings) -> None:
    key = (settings.tessdata_prefix, tuple(settings.ocr_languages))
    if key not in _verified_tessdata:
        verify_tessdata(settings)
        _verified_tessdata.add(key)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(visibility: str, title: str | None) -> str | None:
    if visibility not in _VISIBILITIES:
        raise InvalidInput(f"visibility must be one of {', '.join(_VISIBILITIES)}")
    return title.strip() or None if title else None


def _redact_url(url: str) -> str:
    """URL for logs: no credentials, query string or fragment (they may carry tokens)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path}"


def _url_label(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname or ''}{parts.path}".rstrip("/") or url


def _filename_from_url(url: str, content_type: str) -> str:
    name = unquote(PurePosixPath(urlsplit(url).path).name) or "download"
    if content_type in _TEXT_TYPE_EXT and not PurePosixPath(name).suffix:
        name += _TEXT_TYPE_EXT[content_type]
    return name


def _pick_title(*candidates: str | None) -> str:
    for candidate in candidates:
        if candidate and (clean := normalize_text(candidate).replace("\n", " ")):
            return clean[:_MAX_TITLE]
    return "Untitled"


def _finish_sections(raw: RawExtraction, settings: IngestSettings) -> tuple[list[Section], str]:
    """Blocking part: section, normalize, drop empties, detect languages."""
    sections = []
    for section in build_sections(raw):
        text = normalize_text(section.text)
        if any(ch.isalnum() for ch in text):
            heading = normalize_text(section.heading) if section.heading else None
            sections.append(section.model_copy(update={"text": text, "heading": heading or None}))
    return assign_languages(sections, settings)


async def _document(
    raw: RawExtraction,
    *,
    source: str,
    source_type: SourceType,
    visibility: str,
    title: str | None,
    fallback_title: str,
    content_hash: str | None,
    settings: IngestSettings,
) -> ParsedDocument:
    sections, language = await asyncio.to_thread(_finish_sections, raw, settings)
    if not sections:
        detail = " (OCR found no readable text)" if raw.ocr_used else ""
        raise EmptyDocument(f"no readable text found{detail}")
    text = "\n\n".join(section.text for section in sections)
    return ParsedDocument(
        content_hash=content_hash or _sha256(text.encode()),
        title=_pick_title(
            title,
            raw.metadata_title,
            next((section.heading for section in sections if section.heading), None),
            fallback_title,
        ),
        source=source,
        source_type=source_type,
        mime_type=raw.mime_type,
        visibility=visibility,
        language=language,
        text=text,
        sections=tuple(sections),
        ocr_used=raw.ocr_used,
        parser=raw.parser,
        page_count=raw.page_count,
        warnings=tuple(raw.warnings),
        created_at=datetime.now(UTC),
    )


async def _logged(source: str, run: Callable[[], Awaitable[ParsedDocument]]) -> ParsedDocument:
    start = time.perf_counter()
    try:
        doc = await run()
    except IngestError as exc:
        log.warning(
            "document_rejected",
            source=source,
            error_code=exc.code,
            reason=exc.message,
            duration_ms=round((time.perf_counter() - start) * 1000),
        )
        raise
    log.info(
        "document_preprocessed",
        content_hash=doc.content_hash,
        source=source,
        source_type=doc.source_type,
        mime_type=doc.mime_type,
        parser=doc.parser,
        page_count=doc.page_count,
        sections=len(doc.sections),
        chars=len(doc.text),
        ocr_used=doc.ocr_used,
        language=doc.language,
        duration_ms=round((time.perf_counter() - start) * 1000),
    )
    return doc


async def preprocess_file(
    data: bytes,
    filename: str,
    *,
    visibility: str,
    title: str | None = None,
    settings: IngestSettings | None = None,
) -> ParsedDocument:
    s = settings or get_ingest_settings()
    filename = (filename or "").strip()

    async def run() -> ParsedDocument:
        clean_title = _validate(visibility, title)
        if not filename:
            raise InvalidInput("filename is required")
        if len(data) > s.max_file_bytes:
            raise FileTooLarge(f"file is larger than {s.max_file_mb} MB")
        detected = detect_type(data, filename)
        _verify_tessdata_once(s)
        async with _semaphore(s.max_concurrency):
            raw = await extract_file(data, detected, filename, s)
            return await _document(
                raw,
                source=filename,
                source_type="file",
                visibility=visibility,
                title=clean_title,
                fallback_title=PurePosixPath(filename).stem or filename,
                content_hash=_sha256(data),
                settings=s,
            )

    return await _logged(filename, run)


async def preprocess_url(
    url: str,
    *,
    visibility: str,
    title: str | None = None,
    settings: IngestSettings | None = None,
    client: httpx.AsyncClient | None = None,
    resolve: Resolver = default_resolver,
) -> ParsedDocument:
    s = settings or get_ingest_settings()
    url = (url or "").strip()

    async def run() -> ParsedDocument:
        clean_title = _validate(visibility, title)
        if not url:
            raise InvalidInput("url is required")
        async with _semaphore(s.max_concurrency):
            fetched = await fetch(url, s, client=client, resolve=resolve)
            is_html = fetched.content_type in _HTML_TYPES or (
                not fetched.content_type and b"<html" in fetched.data[:2048].lower()
            )
            if is_html:
                raw = await asyncio.to_thread(extract_html, fetched.data, fetched.final_url)
                content_hash = None  # hash of normalized text: stable across ad/menu changes
                fallback = _url_label(fetched.final_url)
            else:
                name = _filename_from_url(fetched.final_url, fetched.content_type)
                detected = detect_type(fetched.data, name)
                _verify_tessdata_once(s)
                raw = await extract_file(fetched.data, detected, name, s)
                content_hash = _sha256(fetched.data)
                fallback = PurePosixPath(name).stem or _url_label(fetched.final_url)
            return await _document(
                raw,
                source=fetched.final_url,
                source_type="web",
                visibility=visibility,
                title=clean_title,
                fallback_title=fallback,
                content_hash=content_hash,
                settings=s,
            )

    return await _logged(_redact_url(url), run)
