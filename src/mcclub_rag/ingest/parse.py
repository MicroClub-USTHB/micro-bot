"""File bytes -> RawExtraction (Req 2, 3).

Markdown and plain text are decoded in Python. Everything else goes through xberg with
an explicit MIME type (from ``sniff``) and the config from ``ocr.build_xberg_config``.
xberg raises plain ``RuntimeError`` with message prefixes; ``map_xberg_error`` is the
single place that turns those into typed ``IngestError``s.
"""

import asyncio
import re

import charset_normalizer
import xberg

from mcclub_rag.ingest.errors import FileTooLarge, IngestError, ParseError, ParseTimeout
from mcclub_rag.ingest.models import RawExtraction, RawPage
from mcclub_rag.ingest.ocr import build_xberg_config, timeout_budget
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.sniff import DetectedType

_MAX_MESSAGE = 300
_MAX_PAGES = re.compile(r"too many pages: (\d+) \(max: (\d+)\)", re.IGNORECASE)
_OCR_METHODS = (xberg.ExtractionMethod.OCR, xberg.ExtractionMethod.MIXED)
_NOISY_WARNINGS = ("EXIF metadata extraction failed",)


def decode_text(data: bytes) -> str:
    """UTF-8 (BOM tolerated), else the best charset-normalizer guess (cp1256, latin-1, ...)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    best = charset_normalizer.from_bytes(data).best()
    if best is None:
        raise ParseError("undetectable text encoding")
    return str(best)


def map_xberg_error(exc: BaseException) -> IngestError:
    message = str(exc).strip()
    first_line = message.splitlines()[0] if message else f"{type(exc).__name__} during extraction"
    lowered = first_line.lower()

    err: IngestError
    if "too many pages" in lowered:
        found = _MAX_PAGES.search(first_line)
        detail = f"{found.group(1)} pages, limit {found.group(2)}" if found else "over the limit"
        err = FileTooLarge(f"document has too many pages ({detail})")
    elif lowered.startswith("security violation"):
        reason = first_line.split(":", 1)[1].strip()
        err = ParseError(f"rejected by safety limits: {reason}"[:_MAX_MESSAGE])
    elif "timed out" in lowered:
        err = ParseTimeout(f"extraction timed out: {first_line}"[:_MAX_MESSAGE])
    elif "password" in lowered or "encrypt" in lowered:
        err = ParseError("password-protected file")
    else:
        err = ParseError(first_line[:_MAX_MESSAGE])
    err.__cause__ = exc
    return err


def _pdf_page_count(data: bytes) -> int:
    try:
        return xberg.pdf_page_count(data)
    except Exception as exc:  # xberg boundary
        raise map_xberg_error(exc) from exc


def _raw_pages(document: xberg.ExtractedDocument) -> list[RawPage]:
    return [
        RawPage(
            number=page.page_number,
            content=page.content,
            sheet_name=page.sheet_name,
            tables=tuple(
                tuple(tuple(str(cell) for cell in row) for row in table.cells)
                for table in page.tables
            ),
            ocr=page.ocr_confidence is not None,
        )
        for page in document.pages or []
    ]


async def extract_file(
    data: bytes, detected: DetectedType, filename: str, settings: IngestSettings
) -> RawExtraction:
    if detected.kind in ("markdown", "text"):
        return RawExtraction(
            kind=detected.kind,
            mime_type=detected.mime,
            parser="text",
            markdown=decode_text(data),
        )

    page_count: int | None = None
    if detected.kind == "pdf":
        page_count = _pdf_page_count(data)
        if page_count > settings.max_pages:
            raise FileTooLarge(
                f"document has too many pages ({page_count} pages, limit {settings.max_pages})"
            )
    elif detected.kind == "image":
        page_count = 1

    budget = timeout_budget(detected.kind, page_count, settings)
    config = build_xberg_config(settings, budget)
    request = xberg.ExtractInput(
        kind="bytes", bytes=data, mime_type=detected.mime, filename=filename
    )
    try:
        # xberg enforces ``budget`` itself; this outer guard only catches a hung call.
        result = await asyncio.wait_for(xberg.extract(request, config), timeout=budget + 5)
    except TimeoutError as exc:
        raise ParseTimeout(f"extraction exceeded {budget}s") from exc
    except Exception as exc:  # xberg boundary: plain RuntimeError with message prefixes
        raise map_xberg_error(exc) from exc

    if not result.results:
        reason = result.errors[0].message if result.errors else "no result returned"
        raise ParseError(f"extraction failed: {reason}"[:_MAX_MESSAGE])

    document = result.results[0]
    pages = _raw_pages(document)
    return RawExtraction(
        kind=detected.kind,
        mime_type=detected.mime,
        parser=f"xberg {xberg.__version__}",
        markdown=document.content,
        pages=pages,
        metadata_title=(document.metadata.title or "").strip() or None,
        ocr_used=bool(document.metadata.ocr_used or document.extraction_method in _OCR_METHODS),
        page_count=page_count,
        warnings=[
            w.message
            for w in document.processing_warnings
            if not w.message.startswith(_NOISY_WARNINGS)
        ],
    )
