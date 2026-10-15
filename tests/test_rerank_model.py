"""Real default reranker, mmarco int8 (``reranker`` marker): ranking, determinism, latency."""

import time

import pytest

from mcclub_rag.retrieval.registry import DEFAULT_RERANKER, RERANKERS
from mcclub_rag.retrieval.rerank import OnnxReranker

CASES = {
    "ar": (
        "ما هو موعد الجمعية العامة للنادي؟",
        "تنعقد الجمعية العامة للنادي يوم السبت 15 نوفمبر في القاعة الكبرى، ويحضرها جميع الأعضاء.",
        "تعتمد زراعة القمح في السهول على الأمطار الموسمية وجودة التربة.",
    ),
    "fr": (
        "Quel est le montant de la cotisation annuelle ?",
        "La cotisation annuelle des membres est fixée à 1500 dinars, payable en début d'année.",
        "Le Tour de France traverse les Alpes au mois de juillet.",
    ),
    "en": (
        "How do I change the primary color of the theme?",
        "To change the primary color, override the --primary CSS variable in your theme file.",
        "The migration of birds depends on the length of the day and the temperature.",
    ),
}


@pytest.fixture(scope="module")
def reranker(reranker_dir):
    return OnnxReranker(RERANKERS[DEFAULT_RERANKER], reranker_dir)


@pytest.mark.parametrize("lang", sorted(CASES))
def test_relevant_passage_scores_above_unrelated(reranker, lang):
    query, relevant, unrelated = CASES[lang]
    good, bad = reranker.score(query, [relevant, unrelated])
    assert good > bad


def test_score_does_not_depend_on_what_was_scored_before(reranker):
    query, relevant, _ = CASES["fr"]
    alone = reranker.score(query, [relevant])
    others = [text for case in CASES.values() for text in case[1:]]
    assert reranker.score(query, [*others[:5], relevant])[-1] == alone[0]


def test_ten_chunk_sized_pairs_under_ten_seconds(reranker):
    query, relevant, _ = CASES["fr"]
    long_text = " ".join([relevant] * 17)  # about 380 reranker tokens
    start = time.perf_counter()
    reranker.score(query, [long_text] * 10)  # RETRIEVAL_RERANK_TOP_K
    assert time.perf_counter() - start < 10  # CI is ~2x slower than the dev machine
