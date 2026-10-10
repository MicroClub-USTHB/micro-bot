"""RawExtraction -> ordered sections (Req 2.1-2.4).

PDF: per page, split on Markdown headings, the current heading carried across pages, after
repeated header/footer removal. DOCX / Markdown / HTML: split on headings. XLSX: one section
per non-empty sheet, one self-contained "Header: value" line per row so column names survive
chunking. TXT and images: a single section.
"""

import re

from mcclub_rag.ingest.models import RawExtraction, RawPage, Section
from mcclub_rag.text.normalize import strip_repeated_lines

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_ROW_SEP = " · "


def _section(lines: list[str], heading: str | None, page: int | None) -> Section | None:
    text = "\n".join(lines).strip()
    if not any(ch.isalnum() for ch in text):  # empty, or noise such as "#" / "---"
        return None
    return Section(text=text, heading=heading, page=page)


def split_markdown(
    text: str, page: int | None = None, inherited_heading: str | None = None
) -> tuple[list[Section], str | None]:
    """Split at Markdown headings; heading lines stay in their section's text.

    Returns the sections and the last heading seen, so callers can carry it to the next page.
    """
    sections: list[Section] = []
    heading = inherited_heading
    buffer: list[str] = []
    for line in text.split("\n"):
        match = _HEADING.match(line.strip())
        if match:
            if section := _section(buffer, heading, page):
                sections.append(section)
            heading = match.group(2).strip()
            buffer = [line.strip()]
        else:
            buffer.append(line)
    if section := _section(buffer, heading, page):
        sections.append(section)
    return sections, heading


def _row_lines(table: tuple[tuple[str, ...], ...]) -> list[str]:
    rows = [tuple(cell.strip() for cell in row) for row in table]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [_ROW_SEP.join(cell for cell in header if cell)]

    def column(i: int) -> str:
        return header[i] if i < len(header) and header[i] else f"Column {i + 1}"

    return [
        _ROW_SEP.join(f"{column(i)}: {value}" for i, value in enumerate(row) if value)
        for row in body
    ]


def _sheet_section(page: RawPage) -> Section | None:
    lines = [line for table in page.tables for line in _row_lines(table)]
    if not lines:
        # No table cells: keep any prose, minus the "## <sheet>" line xberg injects.
        lines = [
            line
            for line in page.content.split("\n")
            if line.strip() and line.strip() != f"## {page.sheet_name}"
        ]
    return _section(lines, page.sheet_name, None)


def _pdf_sections(pages: list[RawPage]) -> list[Section]:
    contents = strip_repeated_lines([page.content for page in pages])
    sections: list[Section] = []
    heading: str | None = None
    for page, content in zip(pages, contents, strict=True):
        page_sections, heading = split_markdown(
            content, page=page.number, inherited_heading=heading
        )
        sections.extend(page_sections)
    return sections


def build_sections(raw: RawExtraction) -> list[Section]:
    if raw.kind == "pdf" and raw.pages:
        return _pdf_sections(raw.pages)
    if raw.kind == "xlsx":
        return [s for page in raw.pages if (s := _sheet_section(page))]
    if raw.kind == "text":
        return [s] if (s := _section([raw.markdown], None, None)) else []
    if raw.kind == "image":
        return [s] if (s := _section([raw.markdown], None, 1)) else []
    sections, _ = split_markdown(raw.markdown)  # docx, markdown, html, pdf without pages
    return sections
