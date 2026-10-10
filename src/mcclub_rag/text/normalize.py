"""Pure, deterministic text cleanup (Req 5).

``normalize_text`` produces the stored text: faithful to the source, minus invisible noise.
``fold_for_search`` is a lossy variant for the BM25 side only and is never stored.
"""

import math
import re
import unicodedata
from collections import Counter

TATWEEL = "ـ"

# Zero-width / format characters that carry no meaning for search or embedding.
# ZWNJ (U+200C) and ZWJ (U+200D) are deliberately absent: they change Arabic-script shaping.
_INVISIBLE = [
    0x200B,  # zero width space
    0xFEFF,  # BOM / zero width no-break space
    0x2060,  # word joiner
    0x00AD,  # soft hyphen
    0x200E,  # LRM
    0x200F,  # RLM
    *range(0x202A, 0x202F),  # LRE, RLE, PDF, LRO, RLO
    *range(0x2066, 0x206A),  # LRI, RLI, FSI, PDI
]
_CONTROLS = [c for c in range(0x00, 0x20) if c not in (0x09, 0x0A)] + list(range(0x7F, 0xA0))
_DELETE = str.maketrans(dict.fromkeys([*_INVISIBLE, *_CONTROLS, ord(TATWEEL)]))

_NEWLINES = re.compile(r"\r\n|[\r ]")
_PARAGRAPH_SEP = " "
_HSPACE = re.compile(r"[^\S\n]+")  # any Unicode whitespace except newline
_HYPHEN_BREAK = re.compile(r"([^\W\d_])-\n([a-zß-öø-ÿœæ])")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def rejoin_hyphens(text: str) -> str:
    """Join words split by a hyphen at a line break ("infor-\\nmation").

    Only when the next line starts with a lowercase Latin letter, so compounds such as
    "Jean-\\nPierre" and ranges such as "2025-\\n2026" are left alone.
    """
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_PARAGRAPH_SEP, "\n\n")
    text = _NEWLINES.sub("\n", text)
    text = text.translate(_DELETE)
    text = _HSPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = rejoin_hyphens(text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


# --- repeated header / footer removal (Req 5.4) --------------------------------

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def _line_key(line: str) -> str:
    """Normalize a line so "Page 3 / 12" and "Page 4 / 12" compare equal."""
    return _SPACES.sub(" ", _DIGITS.sub("#", line.strip().lower()))


def _is_protected(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("#", "|"))  # Markdown headings and table rows


def strip_repeated_lines(
    pages: list[str], *, edge: int = 2, min_share: float = 0.6, min_pages: int = 3
) -> list[str]:
    """Drop running headers/footers: edge lines that repeat on most pages.

    Only the first and last ``edge`` non-empty lines of each page are candidates, and
    Markdown headings / table rows are never removed. A candidate is removed when its
    digit-masked form appears on at least ``max(min_pages, ceil(min_share * n))`` pages.
    """
    n = len(pages)
    if n < min_pages:
        return list(pages)

    split_pages: list[tuple[list[str], set[int]]] = []
    seen_on_pages: Counter[str] = Counter()
    for page in pages:
        lines = page.split("\n")
        content = [i for i, line in enumerate(lines) if line.strip()]
        edges = {i for i in content[:edge] + content[-edge:] if not _is_protected(lines[i])}
        split_pages.append((lines, edges))
        seen_on_pages.update({_line_key(lines[i]) for i in edges})

    threshold = max(min_pages, math.ceil(min_share * n))
    repeated = {key for key, count in seen_on_pages.items() if count >= threshold}

    return [
        "\n".join(
            line for i, line in enumerate(lines) if not (i in edges and _line_key(line) in repeated)
        )
        for lines, edges in split_pages
    ]


# --- search folding (BM25 side only) -------------------------------------------

_ARABIC_FOLD = str.maketrans({"ٱ": "ا", "ى": "ي", "ة": "ه"})
_DIGIT_FOLD = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def fold_for_search(text: str) -> str:
    """Aggressive, lossy folding for sparse (BM25) matching. Never use for stored text.

    casefold; ٱ→ا, ى→ي, ة→ه; NFKD then drop combining marks, which removes Arabic
    harakat, reduces أ إ آ ؤ ئ to their bare letters and strips Latin accents; remove
    tatweel; Arabic-Indic and Persian digits to ASCII.
    """
    text = text.casefold().translate(_ARABIC_FOLD)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace(TATWEEL, "").translate(_DIGIT_FOLD)
    return unicodedata.normalize("NFC", text)
