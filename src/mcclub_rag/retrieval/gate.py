"""Answer/refuse decision from reranker scores (retrieval Req 5): refusing costs 0 LLM calls."""

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
