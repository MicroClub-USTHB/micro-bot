"""Language detection with py3langid (Req 6).

``detect`` / ``assign_languages`` tag documents and sections for evaluation. They may
return "und" (and "mixed" at document level). ``detect_language`` is the frozen API
for the answer layer and always returns "ar", "fr" or "en".
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from mcclub_rag.ingest.models import Section
from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings

ReplyLanguage = Literal["ar", "fr", "en"]
_REPLY_LANGUAGES: tuple[str, ...] = ("ar", "fr", "en")

# Arabic, Arabic Supplement, Arabic Extended-A, presentation forms A and B
_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


@dataclass(frozen=True)
class LangResult:
    code: str  # ISO 639-1 code, or "und"
    prob: float


@lru_cache(maxsize=4)
def _identifier(codes: tuple[str, ...]) -> LanguageIdentifier:
    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    identifier.set_languages(list(codes))
    return identifier


def _letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def detect(text: str, settings: IngestSettings | None = None) -> LangResult:
    s = settings or get_ingest_settings()
    if _letter_count(text) < s.lang_min_chars:
        return LangResult("und", 0.0)
    code, prob = _identifier(tuple(s.lang_codes)).classify(text)
    if prob < s.lang_min_prob:
        return LangResult("und", float(prob))
    return LangResult(str(code), float(prob))


def assign_languages(
    sections: list[Section], settings: IngestSettings | None = None
) -> tuple[list[Section], str]:
    """Tag each section, then derive the document language weighted by letter count.

    "mixed" when at least two languages each cover ``lang_mixed_share`` of the classified
    letters, "und" when nothing could be classified.
    """
    s = settings or get_ingest_settings()
    tagged: list[Section] = []
    weights: Counter[str] = Counter()
    for section in sections:
        result = detect(section.text, s)
        tagged.append(section.model_copy(update={"language": result.code}))
        if result.code != "und":
            weights[result.code] += _letter_count(section.text)

    total = sum(weights.values())
    if total == 0:
        return tagged, "und"
    major = [code for code, weight in weights.items() if weight / total >= s.lang_mixed_share]
    if len(major) >= 2:
        return tagged, "mixed"
    return tagged, weights.most_common(1)[0][0]


def detect_language(text: str) -> ReplyLanguage:
    """Reply language for a chat message: always "ar", "fr" or "en".

    Arabic-script majority -> "ar" (reliable even for two-word messages); otherwise the
    most probable of ar/fr/en with no threshold; no letters at all -> configured default.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return get_ingest_settings().lang_default
    arabic = sum(1 for ch in letters if _ARABIC_LETTER.match(ch))
    if arabic * 2 > len(letters):
        return "ar"
    code, _ = _identifier(_REPLY_LANGUAGES).classify(text)
    return code
