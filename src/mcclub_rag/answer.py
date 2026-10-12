"""Post-process grounded answers without making model calls.

Model citations use ``[doc_id]``. Models also write ``[doc_id: X]``, the passage label
``[S1]`` or ``[X, Y]``; those are mapped to the passage's ``doc_id`` and rewritten as ``[X]``
(ISSUES.md I22). Citations are validated against the retrieval result.

The fallback (no LLM: no key, budget spent, provider error) never pastes chunks as the answer:
it says so in the user's language and lists the passages as sources (I2, I3, I4).
"""

import re
from collections.abc import Sequence

from mcclub_rag.schemas import Chunk, QueryLang, Source
from mcclub_rag.text.snippet import best_snippet

REFUSAL: dict[str, str] = {
    "en": "I don't know based on the available documents.",
    "fr": "Je ne sais pas d'après les documents disponibles.",
    "ar": "لا أعرف بناءً على الوثائق المتاحة.",
}
UNAVAILABLE: dict[str, str] = {
    "en": "I can't write an answer right now. The most relevant passages are listed in the "
    "sources.",
    "fr": "Je ne peux pas rédiger de réponse pour le moment. Les passages les plus pertinents "
    "sont listés dans les sources.",
    "ar": "لا يمكنني كتابة إجابة الآن. أهم المقاطع ذات الصلة مذكورة في المصادر.",
}
EMPTY_MESSAGE: dict[str, str] = {
    "en": "Please type a question.",
    "fr": "Veuillez écrire une question.",
    "ar": "من فضلك اكتب سؤالا.",
}
SNIPPET_CHARS = 500

_CITATION_RE = re.compile(r"\[\s*(?:doc_id\s*:\s*)?([^\[\]]*?)\s*\]", re.IGNORECASE)
_LABEL_RE = re.compile(r"S(\d+)", re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"\b(?:i\s*(?:don't|do not|don’t)\s+know|je\s+ne\s+sais\s+pas)\b|لا\s+(?:أ|ا)عرف",
    re.IGNORECASE,
)


def _cited_ids(group: str, chunks: Sequence[Chunk]) -> list[str | None]:
    """Doc ids named in one bracket; ``None`` marks an id that cannot be resolved."""
    ids: list[str | None] = []
    for part in (p.strip() for p in group.split(",")):
        label = _LABEL_RE.fullmatch(part)
        if label:
            n = int(label.group(1))
            ids.append(chunks[n - 1].doc_id if 1 <= n <= len(chunks) else None)
        else:
            ids.append(part or None)
    return ids


def build(
    text: str,
    chunks: Sequence[Chunk],
    refused: bool = False,
    query: str = "",
) -> tuple[str, list[Source], bool]:
    """Validate model citations and return the API answer fields.

    Citations outside the retrieval result are removed. A response is grounded
    only when it has one or more valid citations and no invalid citations.
    """
    chunks_by_doc_id = {chunk.doc_id: chunk for chunk in chunks}
    cited: list[str] = []
    has_invalid_citation = False

    def keep_valid_citation(match: re.Match[str]) -> str:
        nonlocal has_invalid_citation
        kept = []
        for doc_id in _cited_ids(match.group(1), chunks):
            if doc_id is None or doc_id not in chunks_by_doc_id:
                has_invalid_citation = True
                continue
            kept.append(f"[{doc_id}]")
            if doc_id not in cited:
                cited.append(doc_id)
        return " ".join(kept)

    answer = _CITATION_RE.sub(keep_valid_citation, text)
    # Snippet around the sentence that matches the question and the answer (names, numbers).
    sources = [_source_from_chunk(chunks_by_doc_id[d], f"{query} {answer}") for d in cited]
    grounded = bool(sources) and not has_invalid_citation and not refused
    if _REFUSAL_RE.search(answer):
        grounded = False
    return answer, sources, grounded


def extractive_fallback(
    chunks: Sequence[Chunk], query: str = "", language: QueryLang = "en"
) -> tuple[str, list[Source], bool]:
    """No LLM: a notice in the user's language, the top two passages as sources, not grounded."""
    selected = list(chunks[:2])
    if not selected:
        return REFUSAL[language], [], False
    return UNAVAILABLE[language], [_source_from_chunk(c, query) for c in selected], False


def _source_from_chunk(chunk: Chunk, query: str = "") -> Source:
    return Source(
        doc_id=chunk.doc_id,
        title=chunk.title,
        snippet=best_snippet(chunk.text, query, max_chars=SNIPPET_CHARS),
    )
