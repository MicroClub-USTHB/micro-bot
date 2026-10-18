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
second call; a paraphrased or translated prompt is not caught.

The fallback (no LLM: no key, budget spent, provider error) never pastes chunks as the answer:
it says so in the user's language and lists the passages as sources (I2, I3, I4).
"""

import re
from collections.abc import Sequence

from micro_bot.messages import NO_ANSWER, UNAVAILABLE
from micro_bot.schemas import Chunk, QueryLang, Source
from micro_bot.text.snippet import best_snippet

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
_LEAK_MIN_CHARS = 25  # the opening "You are Micro Bot, ..." is 45; "Be concise." is too common


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


_OPENERS = "[【［"
_CLOSERS = "]】］"
_CITATION_MAX_CHARS = 80  # an opener left open longer than this is plain text, not a citation
_LEAK_PREFIX_CHARS = 12  # hold a sentence back while it may still become an instruction sentence


class StreamFilter:
    """The model's streamed reply, made safe to show as it arrives (``stream=true``).

    Applies what ``build`` and ``leaks_prompt`` do to a whole reply, one delta at a time:
    a ``NOT_FOUND`` marker is never shown (nor the refusal's citations), ``[S1]`` citations
    become ``[Title]`` once their bracket closes, and text that starts repeating a sentence of
    the instructions or the passage layout is held back, then dropped for good. The streamed
    text is a preview: the final ``done`` event carries the authoritative answer from ``build``.
    """

    def __init__(self, chunks: Sequence[Chunk], system: str) -> None:
        self._chunks = chunks
        self._system = system
        instructions = " ".join(system.split("\nPrior conversation", 1)[0].split()).casefold()
        self._sentences = [
            s for s in _SENTENCE_END_RE.split(instructions) if len(s) >= _LEAK_MIN_CHARS
        ]
        self._raw = ""  # every delta received
        self._pos = -1  # how much of _raw is converted; -1 until the marker check is done
        self._refusal = False
        self._held = ""  # converted text not yet shown
        self._shown = ""  # converted text already shown
        self._stopped = False

    def feed(self, delta: str) -> str:
        """Text to show now for ``delta`` (may be empty: held back or withheld)."""
        self._raw += delta
        return self._emit(final=False)

    def finish(self) -> str:
        """Whatever is still held, once the model has finished."""
        return self._emit(final=True)

    def _emit(self, *, final: bool) -> str:
        if self._stopped:
            return ""
        if self._pos < 0 and not self._check_marker(final):
            return ""
        self._held += self._convert(final)
        if final:
            ready, self._held = self._held, ""
        else:
            cut = max(self._held.rfind(" "), self._held.rfind("\n"))
            if cut < 0:
                return ""
            ready, self._held = self._held[: cut + 1], self._held[cut + 1 :]
            if self._may_leak(ready + self._held):
                self._held = ready + self._held
                return ""
        if leaks_prompt(self._shown + ready, self._system):
            self._stopped = True
            return ""
        self._shown += ready
        return ready

    def _check_marker(self, final: bool) -> bool:
        """Settle whether the reply is a ``NOT_FOUND`` refusal; False while it can't be told."""
        head = re.sub(r"^\W+", "", self._raw)
        if not final and len(head) < len(NOT_FOUND) and NOT_FOUND.startswith(head):
            return False
        marker = _NOT_FOUND_RE.match(self._raw)
        self._refusal = marker is not None
        end = marker.end() if marker else 0
        while end and self._raw[end - 1] in _OPENERS:  # ``\W*`` took a citation's bracket
            end -= 1
        self._pos = end
        return True

    def _convert(self, final: bool) -> str:
        out: list[str] = []
        raw, i = self._raw, self._pos
        while i < len(raw):
            char = raw[i]
            if char in _OPENERS:
                close = next((j for j in range(i + 1, len(raw)) if raw[j] in _CLOSERS), -1)
                if close < 0:
                    if not final and len(raw) - i <= _CITATION_MAX_CHARS:
                        break  # wait for the bracket to close
                    out.append(char)
                    i += 1
                    continue
                if not self._refusal:
                    out.append(self._titles(raw[i + 1 : close]))
                i = close + 1
                continue
            out.append(char)
            i += 1
        self._pos = i
        return "".join(out)

    def _titles(self, group: str) -> str:
        group = re.sub(r"^\s*doc_id\s*:\s*", "", group, flags=re.IGNORECASE).strip()
        doc_ids: list[str] = []
        titles: list[str] = []
        for index in _cited_chunks(group, self._chunks):
            if index is not None and self._chunks[index].doc_id not in doc_ids:
                doc_ids.append(self._chunks[index].doc_id)
                titles.append(_title(self._chunks[index]))
        return f"[{'; '.join(titles)}]" if titles else ""

    def _may_leak(self, text: str) -> bool:
        """The unfinished sentence (or a ``title:`` line) could still turn into a leak."""
        line = text.rsplit("\n", 1)[-1]
        if re.match(r"^\W*title:", line, re.IGNORECASE):
            return True
        sentence = _SENTENCE_END_RE.split(" ".join((self._shown + text).split()))[-1].casefold()
        return len(sentence) >= _LEAK_PREFIX_CHARS and any(
            s.startswith(sentence) for s in self._sentences
        )
