"""History-aware search query without an LLM rewrite (retrieval Req 6, eval-fixes Req 11).

A short follow-up ("et son programme ?") carries no topic of its own, so it is searched together
with the previous question. A short *new* question ("Who is the dean?") is not: only a message
with a follow-up cue (leading conjunction, pronoun, demonstrative, or a bare question word) is
merged. Only the last turn is used: older turns drift to old topics.
"""

import re
from collections.abc import Sequence

from mcclub_rag.text.normalize import fold_for_search, normalize_text, strip_arabic_marks


def _folded(words: str) -> frozenset[str]:
    return frozenset(fold_for_search(words).split())


# Matched on folded word tokens, so accents, hamza forms and case do not matter.
_LEADING = _folded("and also so then et aussi alors puis quid و ثم")
_LEADING_PAIRS = {("what", "about"), ("how", "about"), ("et", "pour"), ("و", "ماذا")}
# Pronouns and demonstratives that point back to the previous turn. Left out on purpose: هو /
# هي ("من هو" = who is), ce ("qu'est-ce"), en ("en quelle année"), there ("is there"), and
# Arabic suffix pronouns (ـه, ـها: too many nouns end in ه).
_POINTERS = _folded(
    "it its they them their he him his she her this that these those "
    "il elle ils elles son sa ses leur leurs lui cet cette ça cela celui celle ceux celles "
    "ذلك تلك هذا هذه هؤلاء هم هما فيه فيها عليه عليها عنه عنها له لها لهم منه منها "
    "تاعو تاعها تاعهم ديالو ديالها ديالهم dyalo dyalha dyalhom ta3o ta3ha ta3hom"
)
# A message made only of these ("Where?", "وين؟", "Et quand ?") is an ellipsis.
_QUESTION_WORDS = _folded(
    "where when who how why what which où quand qui comment pourquoi combien quoi quel quelle "
    "أين متى من كيف لماذا ماذا ما كم هل وين فين شكون علاش وقتاش شحال واش "
    "win fin chkoun 3lach wqtach chhal"
)
_IMPERSONAL = re.compile(r"\by\W+a\W+t\W+il\b")  # "y a-t-il": impersonal, not a pointer
_TOKEN = re.compile(r"\w+")
_REPEATED_PUNCT = re.compile(r"([?!؟.,;:])[?!؟,;:]+")  # "??!!" -> "?"; "..." kept
_HSPACE = re.compile(r"[^\S\n]+")
_MIN_SHOUT_LETTERS = 6
_LATIN_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_CAMEL = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]{2,})+")  # GitHub, McDonald; not "WhEn"


def _clean(text: str) -> str:
    return " ".join(normalize_text(text).split())


def has_followup_cue(message: str) -> bool:
    """True when a message points back to the previous turn. Pure, ar/fr/en + darija."""
    if message.lstrip().startswith(("...", "…")):
        return True
    words = _TOKEN.findall(_IMPERSONAL.sub(" ", fold_for_search(message)))
    if not words:
        return False
    first = words[0]
    if first in _LEADING or tuple(words[:2]) in _LEADING_PAIRS:
        return True
    rest = first[1:]  # Arabic "و" glued to a question word or pointer: وكم, وأين, وشكون
    if first.startswith("و") and (rest in _QUESTION_WORDS or rest in _POINTERS):
        return True
    if any(w in _POINTERS for w in words):
        return True
    return len(words) <= 2 and all(w in _QUESTION_WORDS | _LEADING for w in words)


def build_search_query(
    message: str, history: Sequence[object], max_words: int, *, cues: bool = True
) -> tuple[str, bool]:
    """``(query, used_history)``. Pure; ``history`` items that are not turns are ignored.

    ``cues=False`` merges every short message (the rule before eval-fixes Req 11).
    """
    query = _clean(message)
    if not query or not history or len(query.split()) >= max_words:
        return query, False
    if cues and not has_followup_cue(query):
        return query, False
    previous = getattr(history[-1], "user_message", None)
    if not isinstance(previous, str) or not (previous := _clean(previous)):
        return query, False
    return f"{previous}\n{query}", True


def _fix_word_case(match: re.Match[str]) -> str:
    """Lower-case a word with scrambled case ("wOrkshop"); keep "USTHB", "Python", "GitHub".

    "iPhone" becomes "iphone": harmless for the models, and random noise has that shape too.
    """
    word = match.group()
    if word.islower() or word.isupper() or word.istitle():
        return word
    if _CAMEL.fullmatch(word):
        return word
    return word.lower()


def normalize_query(query: str, fix_case: bool, strip_harakat: bool) -> str:
    """Query sent to the dense encoder and the reranker (eval-fixes Req 9). Pure.

    Repeated punctuation is always collapsed. ``fix_case`` lower-cases a query written mostly in
    capitals and words with scrambled case (acronyms, names and CamelCase are kept);
    ``strip_harakat`` removes Arabic diacritics, since the corpus is mostly unvowelled. BM25
    folds on its own.
    """
    query = _HSPACE.sub(" ", _REPEATED_PUNCT.sub(r"\1", query)).strip()  # keeps the "\n"
    if fix_case:
        cased = [ch for ch in query if ch.isupper() or ch.islower()]
        upper = sum(ch.isupper() for ch in cased)
        if len(cased) >= _MIN_SHOUT_LETTERS and upper >= 0.6 * len(cased):
            query = query.lower()
        else:
            query = _LATIN_WORD.sub(_fix_word_case, query)
    return strip_arabic_marks(query) if strip_harakat else query
