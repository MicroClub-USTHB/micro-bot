"""Deterministic question variants for the robustness suite (Req 6.1).

Each variant keeps the base case's expectations and gets ``id = <base>~<kind>``, so it lands in
the same dev/test split. Randomness is seeded per (seed, case id, kind): reruns give the same
questions. A kind that would not change the question (no accents to strip, say) yields nothing.
"""

import random
import re
import unicodedata
from collections.abc import Iterator

from eval.e2e.cases import Case

KINDS = ("typo", "diacritics", "noise", "filler")
_WORD = re.compile(r"\S+")
_PLAIN = re.compile(r"^[^\W\d_]{4,}[?!.,;:]*$")  # letters only: no digits, ids, urls, #hex
_HARAKAT = re.compile("[ً-ْـ]")  # tanween..sukun, tatweel
_ARABIC_LETTER = re.compile("[ء-ي]")
_FILLER = {
    "en": "please tell me, ",
    "fr": "dis-moi s'il te plaît, ",
    "ar": "من فضلك أخبرني، ",
}


def typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent inner letters, once per 8 letters, in plain words only."""
    words = [m for m in _WORD.finditer(text) if _PLAIN.match(m.group())]
    letters = sum(c.isalpha() for c in text)
    if not words:
        return text
    chars = list(text)
    for match in rng.sample(words, min(len(words), max(1, letters // 8))):
        core = match.group().rstrip("?!.,;:")
        i = match.start() + rng.randrange(1, len(core) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def diacritics(text: str, rng: random.Random) -> str:
    """Arabic: strip harakat and tatweel, or add them to 30% of letters. Latin: strip accents."""
    if _ARABIC_LETTER.search(text):
        if _HARAKAT.search(text):
            return _HARAKAT.sub("", text)
        out = []
        for char in text:
            out.append(char)
            if _ARABIC_LETTER.match(char) and rng.random() < 0.3:
                out.append(rng.choice("َُِ"))
        return "".join(out)
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )


def noise(text: str, rng: random.Random) -> str:
    """Random case in plain words (ids, urls and codes kept), doubled spaces, a trailing '??'."""

    def shout(match: re.Match[str]) -> str:
        word = match.group()
        if not _PLAIN.match(word):
            return word
        return "".join(
            c.swapcase() if c.isascii() and c.isalpha() and rng.random() < 0.3 else c for c in word
        )

    flipped = _WORD.sub(shout, text)
    spaced = "".join("  " if c == " " and rng.random() < 0.3 else c for c in flipped)
    return spaced.rstrip("?") + " ??"


def filler(text: str, lang: str) -> str:
    return _FILLER[lang] + text


def variants(case: Case, seed: int = 0, kinds: tuple[str, ...] = KINDS) -> Iterator[Case]:
    for kind in kinds:
        rng = random.Random(f"{seed}:{case.id}:{kind}")
        if kind == "typo":
            question = typo(case.question, rng)
        elif kind == "diacritics":
            question = diacritics(case.question, rng)
        elif kind == "noise":
            question = noise(case.question, rng)
        else:
            question = filler(case.question, case.lang)
        if question != case.question:
            slices = case.slices if "noisy" in case.slices else [*case.slices, "noisy"]
            yield case.model_copy(
                update={"id": f"{case.id}~{kind}", "question": question, "slices": slices}
            )
