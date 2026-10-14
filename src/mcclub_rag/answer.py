"""Post-process grounded answers without making model calls.

The prompt asks the model to cite passage labels (``[S1]``, ``[S1, S3]``). Models also write a
bare number, ``[doc_id]``, ``[doc_id: X]`` or full-width brackets (``【S1】``, ``［S1］``); all are
accepted (ISSUES.md I22). Citations are validated against the retrieval result, then rewritten
as the cited documents' titles (``[Code of Conduct 2025-2026]``): a person reads which document
backs the sentence, never a label or a uuid. The app writes the title, not the model, so a
title cannot be misspelled or invented.

The model marks a refusal itself: a reply that starts with ``NOT_FOUND`` is not grounded and has
no sources, whatever its wording or language. Matching refusal phrases missed most of them
("Les passages fournis ne précisent pas…", "لا تذكر النصوص…") and left them grounded.

A reply that repeats the system prompt or the passage block (``title:`` / ``text:``) is
withheld (I26): the caller answers with the not-found notice. A string check on the reply, no
second call; a paraphrased or translated prompt is not caught (EDGE_CASES.md).

The fallback (no LLM: no key, budget spent, provider error) never pastes chunks as the answer:
it says so in the user's language and lists the passages as sources (I2, I3, I4).
"""

import re
from collections.abc import Sequence

from mcclub_rag.messages import NO_ANSWER, UNAVAILABLE
from mcclub_rag.schemas import Chunk, QueryLang, Source
from mcclub_rag.text.snippet import best_snippet

NOT_FOUND = "NOT_FOUND"
SNIPPET_CHARS = 500

_CITATION_RE = re.compile(
    r"[\[【［]\s*(?:doc_id\s*:\s*)?([^\[\]【】［］]*?)\s*[\]】］]", re.IGNORECASE
)
_LABEL_RE = re.compile(r"S?(\d+)", re.IGNORECASE)
_REFERENCE_RE = re.compile(r"\s*\[[^\[\]]*\]")
_REPEATED_RE = re.compile(r"(\[[^\[\]]+\])(?:\s*\1)+")
# The marker, with any markdown or punctuation the model wraps around it (``**NOT_FOUND:**``).
_NOT_FOUND_RE = re.compile(rf"^\W*{NOT_FOUND}\W*")
# The passage layout of ``llm/prompts.py``: a fence name, or a "title:" line then "text:".
_PASSAGE_DUMP_RE = re.compile(r"untrusted-source|^\W*title:.*\n\s*text:", re.I | re.M)
_SENTENCE_END_RE = re.compile(r"(?<=[.:])\s+")
_LEAK_MIN_CHARS = 25  # "You are the MC Club assistant." is 30; "Be concise." is too common


def leaks_prompt(reply: str, system: str) -> bool:
    """``reply`` repeats a sentence of the instructions or the passage layout (I26).

    Only the instructions are compared (``system`` up to the prior conversation), whitespace
    collapsed and casefolded, so a reflowed copy still matches.
    """
    if _PASSAGE_DUMP_RE.search(reply):
        return True
    flat = " ".join(reply.split()).casefold()
    instructions = " ".join(system.split("\nPrior conversation", 1)[0].split()).casefold()
    return any(
        len(sentence) >= _LEAK_MIN_CHARS and sentence in flat
        for sentence in _SENTENCE_END_RE.split(instructions)
    )


def strip_references(text: str) -> str:
    """``text`` without its ``[Title]`` references (they name that turn's sources)."""
    return _REFERENCE_RE.sub("", text)


def _cited_chunks(group: str, chunks: Sequence[Chunk]) -> list[int | None]:
    """Chunk indexes named in one bracket; ``None`` marks a citation that cannot be resolved.

    A label or number is that passage; a doc id is the best-ranked chunk of that document.
    """
    indexes: list[int | None] = []
    for part in (p.strip() for p in group.split(",")):
        label = _LABEL_RE.fullmatch(part)
        if label:
            n = int(label.group(1))
            indexes.append(n - 1 if 1 <= n <= len(chunks) else None)
        else:
            found = (i for i, chunk in enumerate(chunks) if part and chunk.doc_id == part)
            indexes.append(next(found, None))
    return indexes


def build(
    text: str,
    chunks: Sequence[Chunk],
    query: str = "",
) -> tuple[str, list[Source], bool]:
    """Validate model citations and return the API answer fields.

    One source per cited document, in order of first citation; its snippet comes from the
    first chunk cited for it. Citations become ``[Title]`` (``[Title A; Title B]`` for several
    documents); citations outside the retrieval result are removed. A response is grounded
    only when it has one or more valid citations and no invalid citations. A ``NOT_FOUND``
    reply loses the marker and its citations; it may come back empty (the caller fills it).
    """
    uncited = _CITATION_RE.sub("", text)
    marker = _NOT_FOUND_RE.match(uncited)
    if marker:
        return re.sub(r"\s+([.,;:!?،؛])", r"\1", uncited[marker.end() :]).strip(), [], False
    cited: dict[str, Chunk] = {}  # doc_id -> chunk the snippet comes from, in citation order
    has_invalid_citation = False

    def to_titles(match: re.Match[str]) -> str:
        nonlocal has_invalid_citation
        doc_ids: list[str] = []
        for index in _cited_chunks(match.group(1), chunks):
            if index is None:
                has_invalid_citation = True
                continue
            chunk = chunks[index]
            cited.setdefault(chunk.doc_id, chunk)
            if chunk.doc_id not in doc_ids:
                doc_ids.append(chunk.doc_id)
        doc_ids.sort(key=list(cited).index)
        return f"[{'; '.join(_title(cited[d]) for d in doc_ids)}]" if doc_ids else ""

    answer = _REPEATED_RE.sub(r"\1", _CITATION_RE.sub(to_titles, text))
    # Snippet around the sentence that matches the question and the answer (names, numbers).
    snippet_query = f"{query} {strip_references(answer)}"
    sources = [_source_from_chunk(chunk, snippet_query) for chunk in cited.values()]
    return answer, sources, bool(sources) and not has_invalid_citation


def _title(chunk: Chunk) -> str:
    """The title as shown inside a citation: brackets would end the citation early."""
    return chunk.title.replace("[", "(").replace("]", ")").strip() or chunk.doc_id


def extractive_fallback(
    chunks: Sequence[Chunk], query: str = "", language: QueryLang = "en"
) -> tuple[str, list[Source], bool]:
    """No LLM: a notice in the user's language, the top two passages as sources, not grounded."""
    selected = list(chunks[:2])
    if not selected:
        return NO_ANSWER[language], [], False
    return UNAVAILABLE[language], [_source_from_chunk(c, query) for c in selected], False


def _source_from_chunk(chunk: Chunk, query: str = "") -> Source:
    return Source(
        doc_id=chunk.doc_id,
        title=chunk.title,
        snippet=best_snippet(chunk.text, query, max_chars=SNIPPET_CHARS),
    )
