import os
import random
from pathlib import Path

import pytest

from mcclub_rag.ingest.errors import FileTooLarge, ParseError, ParseTimeout
from mcclub_rag.ingest.parse import decode_text, extract_file, map_xberg_error
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.ingest.sniff import detect_type

# --- decode_text -----------------------------------------------------------------


def test_decode_utf8_with_bom():
    assert decode_text("﻿Bonjour à tous".encode()) == "Bonjour à tous"


def test_decode_latin1_fallback(fixture_bytes):
    text = decode_text(fixture_bytes("latin1.txt"))
    assert "Règlement intérieur" in text
    assert "présenter leur carte d'adhérent" in text


def test_decode_cp1256_arabic_fallback(fixture_bytes):
    text = decode_text(fixture_bytes("arabic_cp1256.txt"))
    assert "النظام الداخلي للنادي" in text
    assert "بطاقة العضوية" in text


def test_decode_random_bytes_is_parse_error():
    noise = random.Random(7).randbytes(4096)
    with pytest.raises(ParseError, match="encoding"):
        decode_text(noise)


# --- map_xberg_error (exact messages observed from xberg 1.3.6) --------------------


def _mapped(message: str):
    try:
        raise RuntimeError(message)
    except RuntimeError as exc:
        return map_xberg_error(exc), exc


def test_map_too_many_pages():
    err, cause = _mapped(
        "Security violation: Document has too many pages: 2 (max: 1). Raise "
        "`security_limits.max_pages` if this document is legitimate, or split it."
    )
    assert isinstance(err, FileTooLarge)
    assert "too many pages" in err.message and "1" in err.message
    assert err.__cause__ is cause


def test_map_zip_bomb():
    err, _ = _mapped(
        "Security violation: Potential ZIP bomb detected: compressed 4139B -> "
        "uncompressed 17581B (ratio: 4.2:1)"
    )
    assert type(err) is ParseError
    assert err.message.startswith("rejected by safety limits")
    assert "ZIP bomb" in err.message


def test_map_timeout():
    err, _ = _mapped("Extraction timed out after 1ms (limit: 0ms)")
    assert isinstance(err, ParseTimeout)


def test_map_encrypted_pdf():
    err, _ = _mapped(
        "Parsing error: PDF is encrypted and requires a password; set pdf_options.passwords"
    )
    assert type(err) is ParseError
    assert err.message == "password-protected file"


def test_map_generic_parse_error_is_first_line_and_truncated():
    err, cause = _mapped(
        "Parsing error: xberg_native_pdf: failed to load bytes: Invalid cross-reference table"
        + "x" * 1000
        + "\nsecond line with internals"
    )
    assert type(err) is ParseError
    assert err.message.startswith("Parsing error: xberg_native_pdf")
    assert len(err.message) <= 300
    assert "second line" not in err.message
    assert err.__cause__ is cause


def test_map_empty_message():
    err, _ = _mapped("")
    assert type(err) is ParseError and err.message


# --- extract_file with real xberg + fixtures ----------------------------------------


async def _extract(name: str, settings: IngestSettings, fixture_bytes):
    data = fixture_bytes(name)
    return await extract_file(data, detect_type(data, name), name, settings)


async def test_text_pdf_is_native_with_pages(ocr_settings, fixture_bytes):
    raw = await _extract("text.pdf", ocr_settings, fixture_bytes)
    assert raw.kind == "pdf"
    assert raw.ocr_used is False
    assert raw.page_count == 4
    assert [p.number for p in raw.pages] == [1, 2, 3, 4]
    assert "# Annual Fees" in raw.pages[1].content
    assert raw.metadata_title == "MC Club Handbook"
    assert raw.parser.startswith("xberg ")


@pytest.mark.ocr
@pytest.mark.parametrize(
    ("name", "words"),
    [
        ("scanned.pdf", ["ANNUAL FEES REMINDER", "fifty dinars", "end of March"]),
        ("notice.png", ["NOTICE TO ALL MEMBERS", "room B12", "membership card"]),
        ("notice.jpg", ["NOTICE TO ALL MEMBERS", "room B12", "membership card"]),
    ],
)
async def test_ocr_fixtures(ocr_settings, fixture_bytes, name, words):
    raw = await _extract(name, ocr_settings, fixture_bytes)
    assert raw.ocr_used is True
    for word in words:
        assert word in raw.markdown


@pytest.mark.ocr
async def test_arabic_ocr_keeps_every_line_in_reading_order(ocr_settings, fixture_bytes):
    # Regression: xberg's default page segmentation dropped the last two lines.
    raw = await _extract("arabic_notice.png", ocr_settings, fixture_bytes)
    for line_words in ["جميع الأعضاء", "يعقد النادي", "السادسة مساء", "القاعة الكبرى"]:
        assert line_words in raw.markdown
    assert "يدانلا" not in raw.markdown  # "النادي" reversed would mean visual order


@pytest.mark.ocr
async def test_mixed_pdf_ocrs_only_the_scanned_page(ocr_settings, fixture_bytes):
    raw = await _extract("mixed.pdf", ocr_settings, fixture_bytes)
    assert raw.ocr_used is True
    assert [p.ocr for p in raw.pages] == [False, True]
    assert "Club Statutes" in raw.pages[0].content
    assert "ANNUAL FEES REMINDER" in raw.pages[1].content


async def test_docx_keeps_headings_and_table(ocr_settings, fixture_bytes):
    raw = await _extract("doc.docx", ocr_settings, fixture_bytes)
    assert raw.kind == "docx" and raw.ocr_used is False
    assert "# Internal Regulations" in raw.markdown
    assert "## Board" in raw.markdown
    assert "| Chair | Amina |" in raw.markdown


async def test_xlsx_pages_carry_sheet_names_and_cells(ocr_settings, fixture_bytes):
    raw = await _extract("sheet.xlsx", ocr_settings, fixture_bytes)
    assert [p.sheet_name for p in raw.pages] == ["Schedule", "Fees", "Empty"]
    schedule = raw.pages[0].tables[0]
    assert schedule[0] == ("Day", "Activity", "Room")
    assert schedule[1] == ("Monday", "Chess", "B12")


async def test_markdown_and_text_bypass_xberg(ocr_settings, fixture_bytes):
    md = await _extract("notes.md", ocr_settings, fixture_bytes)
    assert md.kind == "markdown" and md.parser == "text"
    assert md.markdown.startswith("# Club Notes")
    txt = await _extract("arabic_cp1256.txt", ocr_settings, fixture_bytes)
    assert txt.kind == "text" and "النظام الداخلي" in txt.markdown


async def test_encrypted_pdf(ocr_settings, fixture_bytes):
    with pytest.raises(ParseError, match="password-protected"):
        await _extract("encrypted.pdf", ocr_settings, fixture_bytes)


async def test_corrupt_pdf(ocr_settings, fixture_bytes):
    with pytest.raises(ParseError):
        await _extract("corrupt.pdf", ocr_settings, fixture_bytes)


async def test_page_limit(tessdata_dir, fixture_bytes):
    settings = IngestSettings(tessdata_prefix=tessdata_dir, max_pages=1)
    with pytest.raises(FileTooLarge, match="pages"):
        await _extract("text.pdf", settings, fixture_bytes)


def _count_files(directory: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(directory)) if directory.exists() else 0


@pytest.mark.ocr
async def test_ocr_writes_no_disk_cache(ocr_settings, fixture_bytes, monkeypatch, tmp_path):
    real_cache = Path.home() / ".cache" / "xberg" / "ocr"
    before = _count_files(real_cache)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    await _extract("notice.png", ocr_settings, fixture_bytes)
    assert _count_files(tmp_path / ".cache" / "xberg" / "ocr") == 0
    assert _count_files(real_cache) == before
