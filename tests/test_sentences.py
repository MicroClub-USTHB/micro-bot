from pathlib import Path

import pytest

from mcclub_rag.ingest.parse import decode_text
from mcclub_rag.text.sentences import split_sentences, split_words

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _stripped(text: str) -> list[str]:
    return [s.strip() for s in split_sentences(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bonjour. Ça va ? Oui !", ["Bonjour.", "Ça va ?", "Oui !"]),
        ("Wait… what? Yes.", ["Wait…", "what?", "Yes."]),
        ("متى الموعد؟ يوم الجمعة.", ["متى الموعد؟", "يوم الجمعة."]),
        ("التسجيل مفتوح؛ لا تتأخروا. شكرا", ["التسجيل مفتوح؛", "لا تتأخروا.", "شكرا"]),
        ("پہلا جملہ۔ دوسرا جملہ۔", ["پہلا جملہ۔", "دوسرا جملہ۔"]),
        ("Il a dit « fini. » Puis il est parti.", ["Il a dit « fini. »", "Puis il est parti."]),
        ('He said "done." Then left.', ['He said "done."', "Then left."]),
        ("(See the annex.) Next point.", ["(See the annex.)", "Next point."]),
        (
            "الاجتماع غدا. Le club ouvre lundi. See you!",
            ["الاجتماع غدا.", "Le club ouvre lundi.", "See you!"],
        ),
    ],
)
def test_boundaries(text, expected):
    assert _stripped(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The budget grew by 3.5 percent this year",
        "Bring a laptop, e.g. a ThinkPad, to the session",
        "Contact M. Dupont before Friday",
        "Install v1.2 of the tool first",
        "Read the docs on example.com before joining",
        "Voir p. 12 du règlement",
        "Talk by Dr. Haddad at noon",
        "Les clubs, etc. sont invités",
        "Le prix est de 1.500 DA",
        "Signed J. K. Rowling yesterday",
    ],
)
def test_non_boundaries(text):
    assert split_sentences(text) == [text]


def test_newline_is_not_a_sentence_boundary_by_itself():
    assert len(split_sentences("first line\nsecond line")) == 1


def test_empty_text():
    assert split_sentences("") == []
    assert split_words("") == []


@pytest.mark.parametrize(
    "text",
    [
        "Bonjour. Ça va ?  Oui !\n\nNouvelle ligne.  ",
        "  leading space. trailing",
        "متى الموعد؟\nيوم الجمعة.  شكرا",
        "a.b.c. d! e?? f",
    ],
)
def test_lossless(text):
    assert "".join(split_sentences(text)) == text
    assert "".join(split_words(text)) == text


@pytest.mark.parametrize("name", ["notes.md", "latin1.txt", "arabic_cp1256.txt"])
def test_lossless_on_fixtures(name):
    text = decode_text((FIXTURES / name).read_bytes())
    pieces = split_sentences(text)
    assert "".join(pieces) == text
    assert len(pieces) >= 1


def test_words_keep_trailing_whitespace():
    assert split_words("un  deux\ttrois") == ["un  ", "deux\t", "trois"]
