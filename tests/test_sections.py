from mcclub_rag.ingest.models import RawExtraction, RawPage
from mcclub_rag.ingest.sections import build_sections, split_markdown


def _raw(kind, markdown="", pages=()):
    return RawExtraction(
        kind=kind, mime_type="x", parser="test", markdown=markdown, pages=list(pages)
    )


def test_split_markdown_on_headings():
    sections, last = split_markdown("Intro line\n\n# Fees\nFifty dinars.\n## Students\nThirty.")
    assert [(s.heading, s.text) for s in sections] == [
        (None, "Intro line"),
        ("Fees", "# Fees\nFifty dinars."),
        ("Students", "## Students\nThirty."),
    ]
    assert last == "Students"


def test_split_markdown_heading_only_section_kept():
    sections, _ = split_markdown("# Title\n\n# Next\nBody")
    assert [s.heading for s in sections] == ["Title", "Next"]


def test_split_markdown_ignores_hash_without_space_and_strips_closing_hashes():
    sections, _ = split_markdown("#hashtag is text\n## Closing ##\nbody")
    assert sections[0].heading is None and sections[0].text == "#hashtag is text"
    assert sections[1].heading == "Closing"


def test_pdf_heading_carried_across_pages_with_page_numbers():
    raw = _raw(
        "pdf",
        pages=[
            RawPage(1, "# Statutes\nArticle one text."),
            RawPage(2, "Continuation of the statutes."),
            RawPage(3, "# Fees\nFifty dinars."),
        ],
    )
    sections = build_sections(raw)
    assert [(s.page, s.heading) for s in sections] == [
        (1, "Statutes"),
        (2, "Statutes"),
        (3, "Fees"),
    ]
    assert sections[1].text == "Continuation of the statutes."


def test_pdf_repeated_footer_removed():
    pages = [
        RawPage(i, f"# Part {i}\nBody sentence number {w}.\nClub MC - Page {i} / 4")
        for i, w in enumerate(["one", "two", "three", "four"], start=1)
    ]
    sections = build_sections(_raw("pdf", pages=pages))
    assert all("Club MC - Page" not in s.text for s in sections)
    assert len(sections) == 4


def test_pdf_without_pages_falls_back_to_markdown():
    sections = build_sections(_raw("pdf", markdown="# A\ntext"))
    assert sections[0].heading == "A" and sections[0].page is None


def test_docx_split_at_headings():
    raw = _raw("docx", markdown="# Rules\nIntro.\n## Board\nSeven members.", pages=[RawPage(1, "")])
    sections = build_sections(raw)
    assert [s.heading for s in sections] == ["Rules", "Board"]
    assert all(s.page is None for s in sections)


def test_xlsx_rows_become_self_contained_lines():
    raw = _raw(
        "xlsx",
        pages=[
            RawPage(
                1,
                "",
                sheet_name="Schedule",
                tables=(
                    (
                        ("Day", "Activity", "Room"),
                        ("Monday", "Chess", "B12"),
                        ("", "", ""),
                        ("Friday", "", "Main hall"),
                    ),
                ),
            ),
            RawPage(2, "## Empty\n\n", sheet_name="Empty"),
            RawPage(3, "", sheet_name="Odd", tables=((("Name", ""), ("Amina", "Chair", "x")),)),
        ],
    )
    sections = build_sections(raw)
    assert [s.heading for s in sections] == ["Schedule", "Odd"]
    assert sections[0].text == (
        "Day: Monday · Activity: Chess · Room: B12\nDay: Friday · Room: Main hall"
    )
    assert sections[1].text == "Name: Amina · Column 2: Chair · Column 3: x"


def test_xlsx_header_only_sheet():
    raw = _raw("xlsx", pages=[RawPage(1, "", sheet_name="S", tables=((("A", "B"),),))])
    assert build_sections(raw)[0].text == "A · B"


def test_markdown_and_html_split_text_single():
    assert [s.heading for s in build_sections(_raw("markdown", "# A\nx\n# B\ny"))] == ["A", "B"]
    assert [s.heading for s in build_sections(_raw("html", "# A\nx\n## B\ny"))] == ["A", "B"]
    sections = build_sections(_raw("text", "# not a heading in txt\nline"))
    assert len(sections) == 1 and sections[0].heading is None


def test_image_is_one_section_on_page_one():
    sections = build_sections(_raw("image", "Line one\n\nLine two"))
    assert len(sections) == 1 and sections[0].page == 1


def test_whitespace_sections_dropped():
    assert build_sections(_raw("markdown", "  \n\n# \n")) == []
    assert build_sections(_raw("image", "   ")) == []
    assert build_sections(_raw("pdf", pages=[RawPage(1, "  "), RawPage(2, "\n")])) == []
    assert build_sections(_raw("image", "--- ... |")) == []
