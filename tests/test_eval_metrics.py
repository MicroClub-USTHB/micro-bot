import pytest

from eval import metrics as m


def test_hit_needs_the_doc_and_a_folded_evidence_string():
    assert m.is_hit("d1", "La Cotisation  annuelle\nest de 1500", "d1", ["cotisation annuelle"])
    assert m.is_hit("d1", "الجمعيّة العامّة للنادي", "d1", ["الجمعية العامة"])  # harakat
    assert m.is_hit("d1", "العـــامة", "d1", ["العامة"])  # tatweel
    assert not m.is_hit("d2", "cotisation annuelle", "d1", ["cotisation annuelle"])
    assert not m.is_hit("d1", "autre chose", "d1", ["cotisation", "vote"])


def test_first_hit_rank_is_one_based():
    ranked = [("d2", "x"), ("d1", "nothing"), ("d1", "the vote")]
    assert m.first_hit_rank(ranked, "d1", ["vote"]) == 3
    assert m.first_hit_rank(ranked, "d1", ["absent"]) is None


def test_hit_at_and_mrr():
    ranks = [1, 3, None, 7]
    assert m.hit_at(ranks, 5) == 0.5
    assert m.mrr_at(ranks, 5) == pytest.approx((1 + 1 / 3) / 4)
    assert m.mrr_at(ranks, 10) == pytest.approx((1 + 1 / 3 + 1 / 7) / 4)
    assert m.mrr_at([None, None], 10) == 0.0
    assert m.hit_at([], 5) == 0.0


def test_sweep_counts_refusals_per_threshold():
    rows = m.sweep(answerable=[0.9, 0.6, 0.3], unanswerable=[0.1, 0.4], step=0.1)
    by_t = {r["threshold"]: r for r in rows}
    assert len(rows) == 11
    assert by_t[0.0] == {"threshold": 0.0, "refused_unanswerable": 0.0, "false_refusal": 0.0}
    assert by_t[0.3]["refused_unanswerable"] == 0.5 and by_t[0.3]["false_refusal"] == 0.0
    assert by_t[0.5]["refused_unanswerable"] == 1.0
    assert by_t[0.5]["false_refusal"] == pytest.approx(1 / 3)


def test_suggestion_is_the_highest_threshold_within_the_budget():
    rows = m.sweep(answerable=[0.9, 0.6, 0.3], unanswerable=[0.1, 0.4], step=0.1)
    assert m.suggest_threshold(rows, 0.0) == 0.3
    assert m.suggest_threshold(rows, 0.34) == 0.6
    assert m.suggest_threshold([], 0.05) is None


def test_access_violations_count_public_rows_with_members_chunks():
    rows = [
        {"role": "public", "visibilities": ["public", "members"]},
        {"role": "public", "visibilities": ["public"]},
        {"role": "member", "visibilities": ["members"]},
        {"role": "public", "visibilities": []},
    ]
    assert m.access_violations(rows) == 1


def test_score_distribution_gives_quartiles():
    dist = m.score_distribution([0.1, 0.2, 0.3, 0.4, 0.5])
    assert dist == {"n": 5, "min": 0.1, "q1": 0.2, "median": 0.3, "q3": 0.4, "max": 0.5}
    assert m.score_distribution([0.7])["q1"] == 0.7
    assert m.score_distribution([]) == {"n": 0}


def test_score_overlap_is_the_share_of_refuse_scores_above_answerable_q1():
    answerable = [0.3, 0.5, 0.7, 0.9, 0.95]  # q1 = 0.5
    assert m.score_overlap(answerable, [0.1, 0.2]) == 0.0
    assert m.score_overlap(answerable, [0.6, 0.9]) == 1.0
    assert m.score_overlap(answerable, [0.5, 0.4, 0.1, 0.8]) == 0.5
    assert m.score_overlap([], [0.5]) is None
    assert m.score_overlap([0.5], []) is None
