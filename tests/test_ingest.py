"""End-to-end: every fixture through preprocess_file, plus error-coverage bookkeeping."""

import importlib

import pytest

from mcclub_rag.ingest import errors
from mcclub_rag.ingest.preprocess import preprocess_file
from tests.conftest import FIXTURES

# name -> (language, ocr_used, minimum sections)
EXPECTED = {
    "text.pdf": ("en", False, 4),
    "scanned.pdf": ("en", True, 1),
    "mixed.pdf": ("en", True, 2),
    "doc.docx": ("en", False, 3),
    "sheet.xlsx": ("en", False, 2),
    "notes.md": ("en", False, 3),
    "latin1.txt": ("fr", False, 1),
    "arabic_cp1256.txt": ("ar", False, 1),
    "notice.png": ("en", True, 1),
    "notice.jpg": ("en", True, 1),
    "arabic_notice.png": ("ar", True, 1),
}
ERROR_FIXTURES = {"encrypted.pdf": errors.ParseError, "corrupt.pdf": errors.ParseError}
# Real scans are added by hand (see tests/fixtures/README.md); skipped until present.
REAL_SCANS = ["arabic_scan.pdf", "arabic_photo.jpg"]


def test_every_fixture_is_covered():
    on_disk = {p.name for p in FIXTURES.iterdir() if p.is_file()} - {
        "README.md",
        "make_fixtures.py",
    }
    assert on_disk == set(EXPECTED) | set(ERROR_FIXTURES)


@pytest.mark.ocr
@pytest.mark.parametrize("name", sorted(EXPECTED))
async def test_fixture_end_to_end(ocr_settings, fixture_bytes, name):
    language, ocr_used, min_sections = EXPECTED[name]
    doc = await preprocess_file(
        fixture_bytes(name), name, visibility="public", settings=ocr_settings
    )
    assert doc.language == language
    assert doc.ocr_used is ocr_used
    assert len(doc.sections) >= min_sections
    assert doc.text and doc.title
    assert all(section.text.strip() for section in doc.sections)


@pytest.mark.parametrize(("name", "error"), sorted(ERROR_FIXTURES.items()))
async def test_bad_fixtures_raise(ocr_settings, fixture_bytes, name, error):
    with pytest.raises(error):
        await preprocess_file(fixture_bytes(name), name, visibility="public", settings=ocr_settings)


@pytest.mark.ocr
@pytest.mark.parametrize("name", REAL_SCANS)
async def test_real_arabic_scan(ocr_settings, name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"real scan {name} not yet provided (see tests/fixtures/README.md)")
    doc = await preprocess_file(path.read_bytes(), name, visibility="public", settings=ocr_settings)
    assert doc.ocr_used is True
    assert doc.language in {"ar", "mixed"}
    assert any("؀" <= ch <= "ۿ" for ch in doc.text)


# Every IngestError subclass must be raised by at least one named test (Req 8.3).
ERROR_TRIGGERS = {
    errors.InvalidInput: "tests.test_preprocess_file:test_invalid_visibility",
    errors.FileTooLarge: "tests.test_preprocess_file:test_oversize_rejected_before_parsing",
    errors.UnsupportedFileType: "tests.test_preprocess_file:test_unsupported_type",
    errors.ParseError: "tests.test_parse:test_corrupt_pdf",
    errors.ParseTimeout: "tests.test_parse:test_map_timeout",
    errors.EmptyDocument: "tests.test_preprocess_file:test_empty_text_file",
    errors.FetchError: "tests.test_web:test_http_error_status",
    errors.UrlNotAllowed: "tests.test_url_guard:test_internal_addresses_rejected",
    errors.OcrConfigError: "tests.test_preprocess_file:test_missing_tessdata_fails_fast",
    errors.TokenizerConfigError: "tests.test_tokens:test_missing_tokenizer_raises_with_instruction",
    errors.EmbedderConfigError: "tests.test_embed:test_missing_stopword_file_fails",
    errors.EmbeddingError: "tests.test_embed:test_inference_failure_becomes_embedding_error",
    errors.IndexUnavailable: "tests.test_store:test_transport_error_becomes_index_unavailable",
    errors.IndexStale: "tests.test_store:test_real_collection_named_like_the_alias_is_refused",
    errors.RerankError: "tests.test_rerank:test_runtime_failure_becomes_rerank_error",
    errors.RerankerConfigError: "tests.test_rerank:test_bad_files_fail_with_download_command",
}

# Pipeline errors whose triggering tests land with later tasks of specs/ingest-pipeline/tasks.md.
# Each entry moves into ERROR_TRIGGERS when its test exists; the set is empty after task 6.8.
PENDING_TRIGGERS: set[type[errors.IngestError]] = set()


def test_every_error_has_a_triggering_test():
    assert not set(ERROR_TRIGGERS) & PENDING_TRIGGERS
    assert set(ERROR_TRIGGERS) | PENDING_TRIGGERS == set(errors.ALL_ERRORS)
    for target in ERROR_TRIGGERS.values():
        module_name, func_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), func_name)), target
