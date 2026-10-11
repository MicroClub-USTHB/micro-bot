"""In-memory exact-match cache for chat answers."""

import unicodedata
from collections.abc import Callable
from typing import TypeVar

from mcclub_rag.schemas import Role

T = TypeVar("T")
CorpusVersion = str | int
CacheKey = tuple[str, Role, CorpusVersion]

_ARABIC_DIACRITICS = frozenset(
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061a"
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f"
    "\u0670\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e7\u06e8\u06ea\u06eb\u06ec\u06ed"
)
_ARABIC_TRANSLATION = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": ""}
)
_GREETINGS = (
    "السلام عليكم ورحمة الله وبركاته",
    "السلام عليكم",
    "مرحبا",
    "اهلا",
    "hello",
    "bonjour",
    "bonsoir",
    "salut",
    "hey",
    "hi",
)


def normalize_arabic(text: str) -> str:
    """Apply the local Arabic fallback until nisbenz supplies its normalizer."""
    translated = text.translate(_ARABIC_TRANSLATION)
    return "".join(character for character in translated if character not in _ARABIC_DIACRITICS)


def normalize(message: str, arabic_normalizer: Callable[[str], str] = normalize_arabic) -> str:
    """Produce an exact-cache key fragment without semantic matching."""
    normalized = arabic_normalizer(message.casefold())
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    normalized = " ".join(normalized.split())
    for greeting in _GREETINGS:
        if normalized == greeting:
            return ""
        if normalized.startswith(f"{greeting} "):
            return normalized.removeprefix(greeting).strip()
    return normalized


class ExactMatchCache[T]:
    """Caches values only for requests that do not carry conversation history."""

    def __init__(
        self,
        corpus_version: Callable[[], CorpusVersion],
        normalizer: Callable[[str], str] = normalize,
    ) -> None:
        self._corpus_version = corpus_version
        self._normalizer = normalizer
        self._entries: dict[CacheKey, T] = {}

    def get(self, message: str, role: Role, *, history: object | None = None) -> T | None:
        if history:
            return None
        return self._entries.get(self._key(message, role))

    def set(self, message: str, role: Role, value: T, *, history: object | None = None) -> None:
        if history:
            return
        self._entries[self._key(message, role)] = value

    def _key(self, message: str, role: Role) -> CacheKey:
        return (self._normalizer(message), role, self._corpus_version())
