"""Snippet around the best-matching sentence (eval-fixes Req 6)."""

from mcclub_rag.text.snippet import best_snippet

FILLER = "Le club organise des activités variées tout au long de l'année universitaire. " * 12
EVIDENCE = "La cotisation annuelle est de 500 DA, payable avant le 15 octobre. "
CHUNK = FILLER + EVIDENCE + FILLER


def test_short_text_is_returned_whole():
    assert best_snippet("Court texte.", "cotisation") == "Court texte."


def test_the_evidence_sentence_in_the_middle_is_shown():
    snippet = best_snippet(CHUNK, "Combien coûte la cotisation annuelle ?")
    assert EVIDENCE.strip() in snippet
    assert len(snippet) <= 500
    assert snippet.startswith("…") and snippet.endswith("…")


def test_no_match_keeps_the_start():
    snippet = best_snippet(CHUNK, "zzz qqq")
    assert snippet.startswith("Le club organise") and not snippet.startswith("…")
    assert len(snippet) <= 500


def test_arabic_question_mark_and_folding():
    text = (
        "مقدمة طويلة عن الكلية. " * 30 + "تفتح المكتبة على الساعة الثامنة صباحا؟ " + "خاتمة. " * 40
    )
    snippet = best_snippet(text, "متى تَفتح المكتبة")
    assert "تفتح المكتبة على الساعة الثامنة" in snippet


def test_tie_goes_to_the_earlier_sentence():
    text = "Alpha cotisation one. " + "x " * 300 + "Beta cotisation two."
    assert "Alpha" in best_snippet(text, "cotisation", max_chars=100)


def test_one_long_sentence_is_cut_to_the_limit():
    text = "word " * 400
    snippet = best_snippet(text, "word", max_chars=120)
    assert len(snippet) <= 120 and snippet.endswith("…")
