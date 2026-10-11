"""Tesseract language data checks and the xberg extraction config (Req 3, 7.1, 7.3).

xberg silently OCRs with fewer languages, or downloads missing traineddata at runtime,
when a language is absent. ``verify_tessdata`` turns that into a hard startup error, and
the config always passes ``tessdata_path`` explicitly.
"""

from pathlib import Path

import xberg

from mcclub_rag.ingest.errors import OcrConfigError
from mcclub_rag.ingest.settings import IngestSettings


def verify_tessdata(settings: IngestSettings) -> Path:
    directory = settings.tessdata_prefix
    missing = []
    for lang in settings.ocr_languages:
        path = directory / f"{lang}.traineddata"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(lang)
    if missing:
        raise OcrConfigError(
            f"Tesseract traineddata missing or empty for {', '.join(missing)} in {directory}. "
            f"Run: python scripts/download_models.py --only tessdata --dest {directory}"
        )
    return directory


def timeout_budget(kind: str, page_count: int | None, settings: IngestSettings) -> int:
    """Seconds allowed for one extraction: a hang guard plus an OCR allowance per page."""
    if kind == "pdf":
        ocr_pages = page_count or 0
    elif kind == "image":
        ocr_pages = 1
    else:
        ocr_pages = 0
    return settings.parse_timeout_s + settings.ocr_page_timeout_s * ocr_pages


def build_xberg_config(settings: IngestSettings, timeout_s: int) -> xberg.ExtractionConfig:
    languages = list(settings.ocr_languages)
    return xberg.ExtractionConfig(
        use_cache=False,
        output_format="markdown",  # headings and tables survive
        pages=xberg.PageConfig(extract_pages=True),  # page numbers, header/footer removal
        ocr_strategy="auto",  # page-level routing: only text-less pages are OCR'd
        ocr_embedded_images=False,
        pdf_options=xberg.PdfConfig(
            extract_images=False, extract_tables=True, extract_annotations=False
        ),
        security_limits=xberg.SecurityLimits(max_pages=settings.max_pages),
        extraction_timeout_secs=timeout_s,
        ocr=xberg.OcrConfig(
            backend="tesseract",
            language=languages,
            tessdata_path=str(settings.tessdata_prefix),
            quality_thresholds=xberg.OcrQualityThresholds(
                min_non_whitespace_per_page=settings.ocr_min_chars_per_page
            ),
            tesseract_config=xberg.TesseractConfig(
                language=languages,
                # Tesseract's own default (fully automatic). xberg's default silently
                # dropped the last lines of an Arabic notice; see test_parse.py.
                psm=3,
                output_format="text",  # table detection garbles plain notices
                enable_table_detection=False,
                use_cache=False,  # otherwise OCR results are cached under ~/.cache/xberg
            ),
        ),
    )
