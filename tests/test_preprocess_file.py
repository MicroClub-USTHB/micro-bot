import pytest
from structlog.testing import capture_logs

from micro_bot.ingest import parse, preprocess
from micro_bot.ingest.errors import (
    EmptyDocument,
    FileTooLarge,
    InvalidInput,
    OcrConfigError,
    UnsupportedFileType,
)
from micro_bot.ingest.preprocess import preprocess_file
from micro_bot.ingest.settings import IngestSettings


async def test_same_bytes_same_content_hash(ocr_settings, fixture_bytes):
    data = fixture_bytes("doc.docx")
    a = await preprocess_file(data, "doc.docx", settings=ocr_settings)
    b = await preprocess_file(data, "renamed.docx", settings=ocr_settings)
    assert a.content_hash == b.content_hash
    assert len(a.content_hash) == 64


async def test_text_pdf_document(ocr_settings, fixture_bytes):
    doc = await preprocess_file(fixture_bytes("text.pdf"), "text.pdf", settings=ocr_settings)
    assert doc.source == "text.pdf" and doc.source_type == "file"
    assert doc.mime_type == "application/pdf"
    assert doc.language == "en"
    assert doc.ocr_used is False
    assert doc.page_count == 4
    assert doc.text == "\n\n".join(s.text for s in doc.sections)
    assert "Club MC - Page" not in doc.text  # repeated footer stripped
    assert {s.page for s in doc.sections} == {1, 2, 3, 4}
    assert [s.heading for s in doc.sections if s.page == 2] == ["Annual Fees"]


@pytest.mark.parametrize(
    ("title", "name", "expected"),
    [
        ("  Given Title ", "text.pdf", "Given Title"),  # caller
        (None, "text.pdf", "MC Club Handbook"),  # document metadata
        (None, "doc.docx", "Internal Regulations"),  # first heading
        (None, "mixed.pdf", "Club Statutes"),  # first heading, past a heading-less section
        (None, "latin1.txt", "latin1"),  # filename stem
    ],
)
async def test_title_fallback_order(ocr_settings, fixture_bytes, title, name, expected):
    doc = await preprocess_file(fixture_bytes(name), name, title=title, settings=ocr_settings)
    assert doc.title == expected


async def test_missing_filename(ocr_settings):
    with pytest.raises(InvalidInput, match="filename"):
        await preprocess_file(b"hello", "  ", settings=ocr_settings)


async def test_oversize_rejected_before_parsing(tessdata_dir, monkeypatch):
    called = []
    monkeypatch.setattr(parse.xberg, "extract", lambda *a, **k: called.append(1))
    settings = IngestSettings(tessdata_prefix=tessdata_dir, max_file_mb=1)
    with pytest.raises(FileTooLarge, match="1 MB"):
        await preprocess_file(b"%PDF-1.4" + b"0" * (1024 * 1024), "big.pdf", settings=settings)
    assert called == []


async def test_unsupported_type(ocr_settings):
    with pytest.raises(UnsupportedFileType):
        await preprocess_file(b"MZ\x90\x00binary", "x.exe", settings=ocr_settings)


@pytest.mark.parametrize("content", [b"", b"   \n\n  ", b"--- | ---"])
async def test_empty_text_file(ocr_settings, content):
    with pytest.raises(EmptyDocument):
        await preprocess_file(content, "empty.txt", settings=ocr_settings)


async def test_missing_tessdata_fails_fast(tmp_path):
    preprocess._verified_tessdata.clear()
    settings = IngestSettings(tessdata_prefix=tmp_path)
    with pytest.raises(OcrConfigError, match="ara"):
        await preprocess_file(b"# Hi\nthere", "a.md", settings=settings)


async def test_success_log_has_metadata_but_no_text(ocr_settings, fixture_bytes):
    with capture_logs() as logs:
        doc = await preprocess_file(fixture_bytes("notes.md"), "notes.md", settings=ocr_settings)
    [event] = [e for e in logs if e["event"] == "document_preprocessed"]
    assert event["content_hash"] == doc.content_hash
    assert event["source"] == "notes.md"
    assert event["sections"] == len(doc.sections)
    assert isinstance(event["duration_ms"], int)
    for section in doc.sections:
        assert all(section.text not in str(value) for value in event.values())
    assert "office is open" not in repr(logs)


async def test_rejection_is_logged_with_code(ocr_settings):
    with capture_logs() as logs, pytest.raises(UnsupportedFileType):
        await preprocess_file(b"MZ", "x.exe", settings=ocr_settings)
    [event] = [e for e in logs if e["event"] == "document_rejected"]
    assert event["error_code"] == "unsupported_file_type"
    assert event["source"] == "x.exe"
