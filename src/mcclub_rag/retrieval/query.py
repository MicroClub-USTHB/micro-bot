"""History-aware search query without an LLM rewrite (retrieval Req 6).

A short follow-up ("et son programme ?") carries no topic of its own, so it is searched together
with the previous question. Only the last turn is used: older turns drift to old topics.
"""

from collections.abc import Sequence

from mcclub_rag.text.normalize import normalize_text


def _clean(text: str) -> str:
    return " ".join(normalize_text(text).split())


def build_search_query(message: str, history: Sequence[object], max_words: int) -> tuple[str, bool]:
    """``(query, used_history)``. Pure; ``history`` items that are not turns are ignored."""
    query = _clean(message)
    if not query or not history or len(query.split()) >= max_words:
        return query, False
    previous = getattr(history[-1], "user_message", None)
    if not isinstance(previous, str) or not (previous := _clean(previous)):
        return query, False
    return f"{previous}\n{query}", True
