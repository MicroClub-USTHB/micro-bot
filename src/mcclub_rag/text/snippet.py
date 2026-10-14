"""Source snippet built around the sentence that best matches the question (eval-fixes Req 6).

The API's ``Source.snippet`` was the chunk's first 500 characters, which often hid the
supporting sentence (ISSUES.md I6, I24). Pure and model-free: sentences from
``text/sentences.split_sentences`` (and line breaks), scored by the overlap of folded words.
"""

import re

from mcclub_rag.text.normalize import fold_for_search
from mcclub_rag.text.sentences import split_sentences

ELLIPSIS = "…"
_WORD = re.compile(r"\w+")


def _words(text: str) -> set[str]:
    """Folded words of 3+ characters, or digits: drops "le", "of", "و" and other glue."""
    return {w for w in _WORD.findall(fold_for_search(text)) if len(w) >= 3 or w.isdigit()}


def _pieces(text: str) -> list[str]:
    return [s.strip() for line in text.splitlines() for s in split_sentences(line) if s.strip()]


def best_snippet(text: str, query: str, *, max_chars: int = 500) -> str:
    """At most ``max_chars`` of ``text``: whole sentences around the best match, "…" at cuts."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    pieces = _pieces(text)
    wanted = _words(query)
    scores = [len(_words(p) & wanted) for p in pieces]
    best = max(range(len(pieces)), key=lambda i: (scores[i], -i))
    if scores[best] == 0:
        best = 0  # nothing matches: the start of the chunk, as before
    start = end = best
    length = len(pieces[best])

    def fits(extra: int) -> bool:  # + separators and both ellipses
        return length + extra + (end - start + 1) + 2 <= max_chars

    while True:
        grown = False
        if end + 1 < len(pieces) and fits(len(pieces[end + 1])):
            end += 1
            length += len(pieces[end])
            grown = True
        if start > 0 and fits(len(pieces[start - 1])):
            start -= 1
            length += len(pieces[start])
            grown = True
        if not grown:
            break
    body = " ".join(pieces[start : end + 1])
    head = ELLIPSIS if start > 0 else ""
    tail = ELLIPSIS if end < len(pieces) - 1 else ""
    if len(head) + len(body) + len(tail) > max_chars:  # one sentence longer than the limit
        body = body[: max_chars - len(head) - 1].rstrip()
        tail = ELLIPSIS
    return f"{head}{body}{tail}"
