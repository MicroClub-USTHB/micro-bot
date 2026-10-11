import io
import zipfile

import pytest

from mcclub_rag.ingest.errors import ParseError, UnsupportedFileType
from mcclub_rag.ingest.sniff import DOCX_MIME, XLSX_MIME, detect_type

OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504


def _zip(*names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, "<xml/>")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("data", "filename", "mime", "kind"),
    [
        (b"%PDF-1.7\n...", "a.pdf", "application/pdf", "pdf"),
        (b"\n\n%PDF-1.4 leading junk is allowed", "scan.bin", "application/pdf", "pdf"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "x.png", "image/png", "image"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 20, "x.jpg", "image/jpeg", "image"),
        (b"II*\x00" + b"\x00" * 20, "x.tif", "image/tiff", "image"),
        (b"MM\x00*" + b"\x00" * 20, "x.tiff", "image/tiff", "image"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "x.webp", "image/webp", "image"),
        (_zip("[Content_Types].xml", "word/document.xml"), "a.docx", DOCX_MIME, "docx"),
        (_zip("[Content_Types].xml", "xl/workbook.xml"), "a.xlsx", XLSX_MIME, "xlsx"),
        (b"# Title\n\nBody", "notes.md", "text/markdown", "markdown"),
        (b"# Title", "notes.MARKDOWN", "text/markdown", "markdown"),
        (b"plain words", "readme.txt", "text/plain", "text"),
        (b"", "empty.txt", "text/plain", "text"),
    ],
)
def test_supported_types(data, filename, mime, kind):
    detected = detect_type(data, filename)
    assert (detected.mime, detected.kind) == (mime, kind)


def test_signature_beats_extension():
    # a real PDF named .txt is still a PDF; a DOCX named .pdf is still a DOCX
    assert detect_type(b"%PDF-1.4", "notes.txt").kind == "pdf"
    assert detect_type(_zip("word/document.xml"), "a.pdf").kind == "docx"


@pytest.mark.parametrize(
    ("data", "filename", "named_type"),
    [
        (_zip("foo.txt"), "archive.zip", "application/zip"),
        (_zip("foo.txt"), "fake.docx", "application/zip"),
        (OLE2, "old.doc", "legacy Office"),
        (OLE2, "old.xls", "legacy Office"),
        (b"\x00\x01garbage\xff" * 10, "a.pdf", "application/octet-stream"),
        (b"MZ\x90\x00 executable", "setup.exe", "application/octet-stream"),
        (b"text with a \x00 NUL byte", "notes.txt", "application/octet-stream"),
        (b"<html><body>hi</body></html>", "page.html", "application/octet-stream"),
        (b"GIF89a....", "anim.gif", "application/octet-stream"),
    ],
)
def test_unsupported_types_name_the_detected_type(data, filename, named_type):
    with pytest.raises(UnsupportedFileType) as exc:
        detect_type(data, filename)
    assert named_type in exc.value.message


@pytest.mark.parametrize("filename", ["secret.docx", "secret.XLSX"])
def test_ole2_with_ooxml_extension_is_encrypted(filename):
    with pytest.raises(ParseError, match="password-protected"):
        detect_type(OLE2, filename)


def test_truncated_zip_is_parse_error():
    with pytest.raises(ParseError, match="corrupt"):
        detect_type(_zip("word/document.xml")[:30], "broken.docx")
