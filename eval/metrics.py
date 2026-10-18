"""Pure metric functions for the retrieval eval (Req 10.3, 10.4, 10.7).

A hit is a returned chunk of the expected document whose text contains one of the question's
evidence strings, both folded (``fold_for_search`` + collapsed whitespace), so labels survive
rechunking, diacritics, tatweel and case.
"""

from collections.abc import Iterable, Sequence

from micro_bot.text.normalize import fold_for_search


def fold(text: str) -> str:
    return " ".join(fold_for_search(text).split())


def is_hit(doc_id: str, text: str, expected_doc_id: str, evidence: Sequence[str]) -> bool:
    if doc_id != expected_doc_id:
        return False
    folded = fold(text)
    return any(fold(e) in folded for e in evidence)


def first_hit_rank(
    ranked: Sequence[tuple[str, str]], expected_doc_id: str, evidence: Sequence[str]
) -> int | None:
    """1-based rank of the first hit among ``(doc_id, text)`` pairs, or None."""
    for rank, (doc_id, text) in enumerate(ranked, start=1):
        if is_hit(doc_id, text, expected_doc_id, evidence):
            return rank
    return None


def hit_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def sweep(
    answerable: Sequence[float], unanswerable: Sequence[float], step: float = 0.01
) -> list[dict[str, float]]:
    """For each threshold: share of unanswerable refused and of answerable wrongly refused."""
    rows = []
    for i in range(round(1 / step) + 1):
        t = round(i * step, 4)
        rows.append(
            {
                "threshold": t,
                "refused_unanswerable": _share(s < t for s in unanswerable),
                "false_refusal": _share(s < t for s in answerable),
            }
        )
    return rows


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def suggest_threshold(rows: Sequence[dict[str, float]], max_false_refusal: float) -> float | None:
    """Highest threshold whose false-refusal share stays within the budget (None if none)."""
    ok = [r["threshold"] for r in rows if r["false_refusal"] <= max_false_refusal]
    return max(ok) if ok else None


def _quantile(ordered: Sequence[float], q: float) -> float:
    return ordered[round(q * (len(ordered) - 1))]


def score_distribution(scores: Iterable[float]) -> dict[str, float]:
    """Gate-score spread (nearest-rank quartiles) for one group of questions (fixes Req 3.5)."""
    ordered = sorted(scores)
    if not ordered:
        return {"n": 0}
    return {"n": len(ordered)} | {
        name: _quantile(ordered, q)
        for name, q in (("min", 0), ("q1", 0.25), ("median", 0.5), ("q3", 0.75), ("max", 1))
    }


def score_overlap(answerable: Sequence[float], refuse: Sequence[float]) -> float | None:
    """Share of refuse-expected scores at or above the answerable Q1: what no gate can separate."""
    if not answerable or not refuse:
        return None
    q1 = _quantile(sorted(answerable), 0.25)
    return sum(s >= q1 for s in refuse) / len(refuse)


def auc(good: Sequence[float], bad: Sequence[float]) -> float | None:
    """ROC AUC: chance that a good case scores above a bad one (ties half). 0.5 = no signal."""
    if not good or not bad:
        return None
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    return round(wins / (len(good) * len(bad)), 4)


def best_threshold(good: Sequence[float], bad: Sequence[float]) -> dict[str, float] | None:
    """Gate threshold with the best Youden J (share of good kept + share of bad refused - 1).

    Candidates are the observed scores; ``kept_good`` = good scores >= t, ``caught_bad`` = bad
    scores < t.
    """
    if not good or not bad:
        return None
    best = None
    for t in sorted(set(good) | set(bad)):
        kept = sum(g >= t for g in good) / len(good)
        caught = sum(b < t for b in bad) / len(bad)
        if best is None or kept + caught > best["kept_good"] + best["caught_bad"]:
            best = {"threshold": round(t, 4), "kept_good": round(kept, 4)}
            best["caught_bad"] = round(caught, 4)
    return best
