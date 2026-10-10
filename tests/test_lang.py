import pytest

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.text.lang import assign_languages, detect, detect_language

AR = (
    "يعقد النادي اجتماعه الشهري يوم الاثنين على الساعة السادسة مساء في القاعة الكبرى. "
    "يرجى من جميع الأعضاء الحضور في الموعد وإحضار بطاقة العضوية."
)
FR = (
    "Le club organise sa réunion mensuelle lundi à dix-huit heures dans la grande salle. "
    "Tous les membres sont priés d'arriver à l'heure avec leur carte d'adhérent."
)
EN = (
    "The club holds its monthly meeting on Monday at six in the evening in the main hall. "
    "All members are asked to arrive on time and bring their membership card."
)


@pytest.fixture
def settings():
    return IngestSettings()


@pytest.mark.parametrize(("text", "code"), [(AR, "ar"), (FR, "fr"), (EN, "en")])
def test_detect_paragraphs(settings, text, code):
    result = detect(text, settings)
    assert result.code == code
    assert result.prob >= settings.lang_min_prob


@pytest.mark.parametrize("text", ["ok", "", "123 456", "?!"])
def test_detect_too_short_is_und(settings, text):
    assert detect(text, settings).code == "und"


def test_detect_restricted_to_configured_codes(settings):
    # Spanish is outside the corpus languages; the result must still be one of them or und
    result = detect("La reunión del club será el lunes por la tarde en la sala grande.", settings)
    assert result.code in {"ar", "fr", "en", "und"}


def _sections(*texts):
    return [Section(text=t) for t in texts]


def test_assign_single_language(settings):
    sections, lang = assign_languages(_sections(FR, FR), settings)
    assert lang == "fr"
    assert [s.language for s in sections] == ["fr", "fr"]


def test_assign_mixed_ar_fr(settings):
    sections, lang = assign_languages(_sections(AR, FR), settings)
    assert lang == "mixed"
    assert [s.language for s in sections] == ["ar", "fr"]


def test_minor_language_below_share_is_not_mixed(settings):
    sections, lang = assign_languages(_sections(*[FR] * 19, EN), settings)  # 5% English
    assert lang == "fr"
    assert sections[-1].language == "en"


def test_assign_all_unclassifiable_is_und(settings):
    sections, lang = assign_languages(_sections("", "ok", "12"), settings)
    assert lang == "und"
    assert {s.language for s in sections} == {"und"}


def test_assign_keeps_other_section_fields(settings):
    sections, _ = assign_languages([Section(text=FR, heading="Intro", page=3)], settings)
    assert sections[0].heading == "Intro" and sections[0].page == 3


# --- detect_language: the frozen function the answer layer calls ---------------


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("متى الاجتماع؟", "ar"),
        ("كم ثمن الاشتراك", "ar"),
        ("Quand est la réunion du club ?", "fr"),
        ("Combien coûte l'adhésion annuelle ?", "fr"),
        ("When is the club meeting?", "en"),
        ("How much is the yearly membership fee?", "en"),
        ("متى موعد الاجتماع meeting", "ar"),  # Arabic-script letters are the majority
    ],
)
def test_detect_language(text, code):
    assert detect_language(text) == code


@pytest.mark.parametrize("text", ["123 ?", "", "   ", "!!!"])
def test_detect_language_without_letters_uses_default(text, monkeypatch):
    monkeypatch.setenv("INGEST_LANG_DEFAULT", "en")
    from mcclub_rag.ingest.settings import get_ingest_settings

    get_ingest_settings.cache_clear()
    try:
        assert detect_language(text) == "en"
    finally:
        get_ingest_settings.cache_clear()


@pytest.mark.parametrize("text", ["ok", "salut", "hi", "ok merci", "x", "Bonjour!", "MC"])
def test_detect_language_always_returns_a_corpus_language(text):
    assert detect_language(text) in {"ar", "fr", "en"}
