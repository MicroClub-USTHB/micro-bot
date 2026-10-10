"""Post-process grounded answers without making model calls.

Model citations use ``[doc_id]``. The prompt labels passages ``[S1]`` etc., but
includes each passage's ``doc_id``; models must cite that id rather than a
passage ordinal so citations can be validated against the retrieval result.
"""

import re
from collections.abc import Sequence

from mcclub_rag.schemas import Chunk, Source

_CITATION_RE = re.compile(r"\[([^\[\]]*)\]")
_REFUSAL_RE = re.compile(
    r"\b(?:i\s*(?:don't|do not|don’t)\s+know|je\s+ne\s+sais\s+pas)\b|لا\s+(?:أ|ا)عرف",
    re.IGNORECASE,
)


def build(
    text: str,
    chunks: Sequence[Chunk],
    refused: bool = False,
) -> tuple[str, list[Source], bool]:
    """Validate model citations and return the API answer fields.

    Citations outside the retrieval result are removed. A response is grounded
    only when it has one or more valid citations and no invalid citations.
    """
    chunks_by_doc_id = {chunk.doc_id: chunk for chunk in chunks}
    sources: list[Source] = []
    seen_ids: set[str] = set()
    has_invalid_citation = False

    def keep_valid_citation(match: re.Match[str]) -> str:
        nonlocal has_invalid_citation
        cited_id = match.group(1)
        chunk = chunks_by_doc_id.get(cited_id)
        if not cited_id or chunk is None:
            has_invalid_citation = True
            return ""
        if cited_id not in seen_ids:
            sources.append(_source_from_chunk(chunk))
            seen_ids.add(cited_id)
        return match.group(0)

    answer = _CITATION_RE.sub(keep_valid_citation, text)
    grounded = bool(sources) and not has_invalid_citation and not refused
    if _REFUSAL_RE.search(answer):
        grounded = False
    return answer, sources, grounded


def extractive_fallback(chunks: Sequence[Chunk]) -> tuple[str, list[Source], bool]:
    """Return the top one or two retrieved chunks verbatim without an LLM call."""
    selected = list(chunks[:2])
    sources = [_source_from_chunk(chunk) for chunk in selected]
    answer = "\n\n".join(chunk.text for chunk in selected)
    return answer, sources, bool(selected) and not _REFUSAL_RE.search(answer)


def _source_from_chunk(chunk: Chunk) -> Source:
    return Source(doc_id=chunk.doc_id, title=chunk.title, snippet=chunk.text[:500])
