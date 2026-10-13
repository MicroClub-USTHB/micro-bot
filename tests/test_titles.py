"""PDF title heuristic (eval-fixes Req 14.2, 14.3)."""

import pytest

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.preprocess import is_generic_title, page_one_title


def _sections(*rows):
    return [Section(text="x", heading=h, page=p) for p, h in rows]


# Page-1 sections of the two real PDFs (data/parsed dumps, 2026-10-11)
COC = _sections(
    (1, None),
    (1, "MICRO CLUB"),
    (1, "Code of Conduct"),
    (1, "2025–2026 Season"),
    (2, "Contents"),
    (3, "Chapter I: Introduction"),
)
SG = _sections((1, None), (1, "BOULOUAH Mahdi"), (1, "Lets Experience IT \\!"), (2, "About me"))


def test_page_one_title_of_the_real_pdfs():
    assert page_one_title(COC) == "MICRO CLUB: Code of Conduct 2025–2026 Season"
    assert page_one_title(SG) == "BOULOUAH Mahdi: Lets Experience IT !"


def test_page_one_title_skips_noise_and_duplicates():
    sections = _sections(
        (1, "3"),
        (1, "Sommaire"),
        (1, "12/03/2026"),
        (1, "Statuts du club"),
        (1, "STATUTS DU CLUB"),
        (1, "Ab"),
        (1, "Titre I"),
        (1, "Article 1"),
        (1, "Article 2"),
    )
    assert page_one_title(sections) == "Statuts du club: Titre I Article 1"


def test_no_page_one_heading():
    assert page_one_title(_sections((1, None), (2, "Later"))) is None
    assert page_one_title(_sections((None, "Markdown heading"))) is None


@pytest.mark.parametrize(
    ("title", "generic"),
    [
        ("KENPA SG DOC", True),  # the filename stem
        ("kenpa_sg-doc", True),
        ("MICRO CLUB", True),  # part of the page-1 title
        ("Untitled", True),
        ("Microsoft Word - statuts.docx", True),
        ("Document1", True),
        ("MC Club Handbook", False),
        ("Club Rules 2025", False),
    ],
)
def test_generic_titles(title, generic):
    page_one = "MICRO CLUB: Code of Conduct 2025–2026 Season"
    assert is_generic_title(title, "KENPA SG DOC", page_one) is generic
