"""Answer/refuse decision from reranker scores (retrieval Req 5): refusing costs 0 LLM calls.

The gate is a **relevance floor** (eval-fixes Req 3.4): it refuses when nothing retrieved is about
the question. It cannot tell an answerable question from an in-domain unknown (their scores
overlap, ISSUES.md I8); that refusal is the LLM's job. Calibrate with ``eval/calibrate_gate.py``.
"""

from collections.abc import Sequence

from mcclub_rag.schemas import RetrieveDecision


def decide(scores: Sequence[float], threshold: float | None) -> tuple[float, RetrieveDecision]:
    """``(top_score, decision)``. Pure. ``threshold=None`` = not calibrated: answer if any."""
    if not scores:
        return 0.0, "refuse"
    top = max(scores)
    if threshold is not None and top < threshold:
        return top, "refuse"
    return top, "answer"
