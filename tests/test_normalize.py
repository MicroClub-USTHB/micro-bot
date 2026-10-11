import unicodedata

import pytest

from mcclub_rag.text.normalize import (
    fold_for_search,
    normalize_text,
    rejoin_hyphens,
    strip_repeated_lines,
)

ZWNJ, ZWJ = "‌", "‍"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # line endings
        ("a\r\nb\rc", "a\nb\nc"),
        ("a b", "a\nb"),
        # C0 / C1 controls removed, \n and \t handled
        ("a\x00b\x07c\x1fd\x7fe\x85f", "abcdef"),
        ("a\tb", "a b"),
        # zero-width / format characters removed
        ("in​visible﻿", "invisible"),
        ("soft­hyphen", "softhyphen"),
        ("word⁠joiner", "wordjoiner"),
        # bidi controls and LRM/RLM removed
        ("‫مرحبا‬ ⁧abc⁩ ‎‏x", "مرحبا abc x"),
        # ZWNJ / ZWJ kept (they change Arabic-script shaping)
        (f"می{ZWNJ}خواهم {ZWJ}", f"می{ZWNJ}خواهم {ZWJ}"),
        # tatweel removed
        ("مـــرحبا", "مرحبا"),
        # Unicode spaces -> one space, runs collapsed, lines stripped
        ("a  b c　d", "a b c d"),
        ("  lead   and   trail  \n  next ", "lead and trail\nnext"),
        # paragraph breaks kept, >=3 newlines collapsed to 2
        ("p1\n\n\n\n\np2", "p1\n\np2"),
        ("p1\n \n \np2", "p1\n\np2"),
        ("p1\n\np2", "p1\n\np2"),
        # hyphenation across line breaks
        ("infor-\nmation", "information"),
        ("infor- \n  mation", "information"),
        ("déve-\nloppement", "développement"),
        ("Jean-\nPierre", "Jean-\nPierre"),
        ("page 12-\n13", "page 12-\n13"),
        # NFC
        ("été", "été"),
        # surrounding blank lines trimmed
        ("\n\n  hello \n\n", "hello"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_output_is_nfc():
    out = normalize_text("Café à l'école")
    assert unicodedata.is_normalized("NFC", out)


@pytest.mark.parametrize(
    "raw",
    [
        "infor-\nmation\r\n\r\n\r\nNext para​",
        "مـرحبا‫ بالعالم‬\n\n\n\nسطر",
        "| a | b |\n| --- | --- |\n| 1 | 2 |",
        "",
    ],
)
def test_normalize_is_idempotent(raw):
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_markdown_table_survives():
    table = "| Day | Activity |\n| --- | --- |\n| Monday | Chess |"
    assert normalize_text(table) == table


def test_rejoin_hyphens_only_before_lowercase_letter():
    assert rejoin_hyphens("pré-\nsident et Jean-\nPierre, 2025-\n2026") == (
        "président et Jean-\nPierre, 2025-\n2026"
    )


# --- strip_repeated_lines ---------------------------------------------------


BODIES = [
    ("The board met on Monday.", "Minutes were approved unanimously."),
    ("Fees are due in March.", "Late payments incur a small penalty."),
    ("Elections take place in May.", "Candidates must register a week ahead."),
    ("The library opens at nine.", "Books can be borrowed for two weeks."),
    ("Members may invite guests.", "Guests must sign the visitor book."),
    ("Training starts at six.", "Bring your own equipment."),
    ("The trip is planned for June.", "Seats are limited to forty people."),
]


def _pages(n: int) -> list[str]:
    return [
        f"MC Club Bulletin\n# Article {i}\n{BODIES[i - 1][0]}\n{BODIES[i - 1][1]}\n"
        f"Club MC - Page {i} / {n}"
        for i in range(1, n + 1)
    ]


def test_strip_repeated_header_and_numbered_footer():
    out = strip_repeated_lines(_pages(5))
    for i, page in enumerate(out, start=1):
        assert "MC Club Bulletin" not in page
        assert "Page" not in page
        # numbered headings look alike once digits are masked, but headings are never stripped
        assert f"# Article {i}" in page
        assert BODIES[i - 1][0] in page
        assert BODIES[i - 1][1] in page


def test_line_repeated_on_too_few_pages_is_kept():
    pages = _pages(5)
    pages[0] = "Draft\n" + pages[0]
    pages[1] = "Draft\n" + pages[1]
    out = strip_repeated_lines(pages)
    assert out[0].startswith("Draft") and out[1].startswith("Draft")
    assert all("MC Club Bulletin" not in p for p in out)


def test_fewer_than_three_pages_unchanged():
    pages = ["Header\nA\nFooter", "Header\nB\nFooter"]
    assert strip_repeated_lines(pages) == pages


def test_repeated_mid_page_lines_untouched():
    pages = [
        f"Top {i}\nl1 {i}\nl2 {i}\nl3 {i}\nSee annex B.\nl4 {i}\nl5 {i}\nl6 {i}\nBottom {i}x"
        for i in range(4)
    ]
    out = strip_repeated_lines(pages)
    assert all("See annex B." in p for p in out)


def test_table_rows_never_stripped():
    pages = [f"| Name | Role |\n| --- | --- |\n{a}\n{b}" for a, b in BODIES[:4]]
    assert strip_repeated_lines(pages) == pages


def test_strip_returns_same_page_count():
    assert len(strip_repeated_lines(_pages(7))) == 7


# --- fold_for_search ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أحمد", "احمد"),
        ("إحمد", "احمد"),
        ("آحمد", "احمد"),
        ("ٱلكتاب", "الكتاب"),
        ("مدرسة", "مدرسه"),
        ("مستشفى", "مستشفي"),
        ("مسؤول", "مسوول"),
        ("رئيس", "رييس"),
        ("مُحَمَّدٌ", "محمد"),
        ("مـــرحبا", "مرحبا"),
        ("Événement", "evenement"),
        ("Crème Brûlée", "creme brulee"),
        ("Straße", "strasse"),
        ("٣٤", "34"),
        ("۳۴", "34"),
        ("ﻻ", "لا"),
    ],
)
def test_fold_for_search(raw, expected):
    assert fold_for_search(raw) == expected


@pytest.mark.parametrize("raw", ["أَحْمَد Événement ٣٤", "مستشفى مدرسة", ""])
def test_fold_is_idempotent(raw):
    once = fold_for_search(raw)
    assert fold_for_search(once) == once
