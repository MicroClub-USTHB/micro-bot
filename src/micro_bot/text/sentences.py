"""Lossless multilingual sentence and word splitting (chunking Req 3.3).

Pure regex rules, no model: pysbd has no Arabic rules, and chunking only needs
punctuation-aware boundaries. Every piece keeps its trailing whitespace, so
``"".join(split_sentences(text)) == text``; the chunker relies on that to never lose text.
"""

import re

# A terminator run, optional closing quotes/brackets (French style may have a space before
# "»"), then the whitespace that must follow; the boundary sits after that whitespace.
_BOUNDARY = re.compile(r"""[.!?…؟؛۔]+(?:\s?["'»”’)\]])*(?:\s+|$)""")
_TOKEN_BEFORE = re.compile(r"(\S+)$")
_WORD = re.compile(r"\S+\s*|\s+")

# Lower-cased tokens (without the final dot) that a "." does not end a sentence after.
_ABBREVIATIONS = frozenset(
    {
        "e.g", "i.e", "etc", "cf", "vs", "p", "pp", "n°", "fig", "art", "al",
        "m", "mm", "mme", "mlle", "dr", "pr", "prof", "mr", "mrs", "ms", "st",
    }
)  # fmt: skip


def _is_boundary(text: str, match: re.Match[str]) -> bool:
    if match.end() == len(text):
        return True
    terminators = match.group().rstrip()
    if terminators.rstrip("\"'»”’)] ") != ".":
        return True  # "!", "?", "…", "؟", "؛", "۔" or a run like "?!" always end a sentence
    before = _TOKEN_BEFORE.search(text, 0, match.start())
    token = before.group(1).lstrip("(\"'«“[").lower() if before else ""
    if token in _ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha() and token.isascii():
        return False  # an initial, as in "J. K. Rowling"
    following = text[match.end()]
    return not (following.isdigit() or (following.islower() and following.isascii()))


def split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _is_boundary(text, match):
            pieces.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def split_words(text: str) -> list[str]:
    """Split after whitespace runs; leading whitespace forms its own piece."""
    return _WORD.findall(text)
