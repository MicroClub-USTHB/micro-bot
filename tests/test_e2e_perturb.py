"""Deterministic question variants (Req 6.1): golden outputs, what typo must not touch."""

import random

from eval.e2e.cases import Case
from eval.e2e.perturb import KINDS, diacritics, noise, typo, variants

EN = "How many formal warnings lead to a permanent ban?"
FR = "Quel est le délai des fiches PFE ?"
AR = "ما هو آخر أجل للتسجيل في برنامج أدرس في السعودية؟"


def _case(question: str, lang: str) -> Case:
    return Case(
        id="x", question=question, lang=lang, role="public", expect="refuse", slices=["lookup"]
    )


def test_golden_variants_with_seed_0():
    en = {c.id: c.question for c in variants(_case(EN, "en"))}
    assert en == {
        "x~typo": "How mnay fomral wanrings laed to a permnaent ban?",
        "x~noise": "How Many fOrmal warnings  lEAd to a  pErmaNent  ban ??",
        "x~filler": "please tell me, " + EN,
    }  # no accents to strip: no diacritics variant
    fr = {c.id: c.question for c in variants(_case(FR, "fr"))}
    assert fr["x~typo"] == "Qeul est le dléai des fcihes PFE ?"
    assert fr["x~diacritics"] == "Quel est le delai des fiches PFE ?"
    ar = {c.id: c.question for c in variants(_case(AR, "ar"))}
    assert ar["x~typo"] == "ما هو آخر أجل للستجيل في برنماج أردس في السعودية؟"
    assert diacritics(ar["x~diacritics"], random.Random(0)) == AR  # stripping restores it


def test_variants_keep_expectations_and_add_noisy():
    for variant in variants(_case(EN, "en")):
        assert variant.expect == "refuse" and variant.slices == ["lookup", "noisy"]


def test_typo_and_noise_leave_ids_urls_codes_and_numbers_alone():
    text = "Is X-API-Key pk_66db96db valid at https://mcdi.microclub.info for #001EFF since 1985?"
    keep = ["X-API-Key", "pk_66db96db", "https://mcdi.microclub.info", "#001EFF", "1985"]
    for seed in range(20):
        for changed in (typo(text, random.Random(seed)), noise(text, random.Random(seed))):
            assert all(token in changed for token in keep), changed


def test_reruns_are_identical():
    first = [c.question for c in variants(_case(AR, "ar"), seed=3)]
    assert first == [c.question for c in variants(_case(AR, "ar"), seed=3)]
    assert len(first) == len(KINDS)
