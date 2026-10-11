"""Magic-byte file type detection (Req 2.5).

Runs before xberg, which trusts the file extension when bytes are ambiguous. The
extension is only used to choose between Markdown and plain text, never to override
a binary signature.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from mcclub_rag.ingest.errors import ParseError, UnsupportedFileType
from mcclub_rag.ingest.models import SourceKind

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_MARKDOWN_EXT = {".md", ".markdown"}
_TEXT_EXT = {".txt"}
_HEAD = 8192


@dataclass(frozen=True)
class DetectedType:
    mime: str
    kind: SourceKind


def _unsupported(detected: str) -> UnsupportedFileType:
    return UnsupportedFileType(f"unsupported file type: {detected}")


def _image_mime(head: bytes) -> str | None:
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _zip_type(data: bytes) -> DetectedType:
    try:
        names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())  # central directory only
    except zipfile.BadZipFile as exc:
        raise ParseError("corrupt or truncated ZIP-based file") from exc
    if "word/document.xml" in names:
        return DetectedType(DOCX_MIME, "docx")
    if "xl/workbook.xml" in names:
        return DetectedType(XLSX_MIME, "xlsx")
    raise _unsupported("application/zip")


def detect_type(data: bytes, filename: str) -> DetectedType:
    head = data[:_HEAD]
    ext = PurePath(filename).suffix.lower()

    if b"%PDF-" in data[:1024]:
        return DetectedType("application/pdf", "pdf")
    if mime := _image_mime(head):
        return DetectedType(mime, "image")
    if head.startswith(b"PK\x03\x04"):
        return _zip_type(data)
    if head.startswith(_OLE2):
        # Password-protected .docx/.xlsx are wrapped in an OLE2 container.
        if ext in {".docx", ".xlsx"}:
            raise ParseError("password-protected (encrypted) Office file")
        raise _unsupported("legacy Office (OLE2) file such as .doc or .xls")
    if b"\x00" not in head:
        if ext in _MARKDOWN_EXT:
            return DetectedType("text/markdown", "markdown")
        if ext in _TEXT_EXT:
            return DetectedType("text/plain", "text")
    raise _unsupported("application/octet-stream")
