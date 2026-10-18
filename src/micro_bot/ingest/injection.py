"""Flag text that gives instructions to an AI assistant (eval-fixes Req 4.3, ISSUES.md I3/I11).

Defence in depth only: the prompt already fences passages as untrusted data, and the answer layer
must never echo chunks. The flag is stored in the chunk payload so retrieval can down-weight such
chunks (``RETRIEVAL_INSTRUCTION_PENALTY``). Stored text is never changed. Pure, deterministic,
matched on ``fold_for_search`` text (no accents, bare hamza, ة -> ه, ى -> ي, lower case).
"""

import re

from micro_bot.text.normalize import fold_for_search

_PATTERNS = [
    # en
    r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}"
    r"\b(instructions?|prompts?|rules)\b",
    r"\b(reveal|print|show|repeat|output|leak)\b.{0,20}\b(your|the)\b.{0,10}"
    r"\b(system prompt|instructions|hidden prompt)\b",
    r"\binstructions? for (ai|the|any) (assistants?|models?|chatbots?|bots?)\b",
    r"\bfrom now on\b.{0,10}\b(answer|reply|respond|end|always|only|say|write)\b",
    r"\byou are now\b.{0,40}\b(assistant|ai|model|bot|dan|jailbroken|unrestricted|without)\b",
    # fr
    r"\bignore[rz]?\b.{0,30}\binstructions?\b.{0,20}\b(precedentes?|anterieures?|ci dessus)\b",
    r"\b(revele|revelez|affiche|affichez|montre|montrez|donne|donnez)\b.{0,15}"
    r"\b(ton|votre|le)\b.{0,10}\b(prompt|invite|message) systeme\b",
    r"\binstructions? pour (l ?)?(ia|assistant|assistants|modele|chatbot)\b",
    r"\ba partir de maintenant\b.{0,10}\b(reponds|repondez|ecris|ecrivez|dis|dites)\b",
    r"\btu es (maintenant|desormais)\b",
    # ar
    r"تجاهل.{0,20}(التعليمات|الاوامر|قواعدك|تعليماتك)",
    r"تعليمات (للمساعد|للذكاء|للنموذج|للروبوت)",
    r"(اكشف|اظهر|اعرض|اطبع).{0,20}(تعليمات|موجه|اوامر) النظام",
    r"من الان فصاعدا.{0,20}(اكتب|اجب|رد|قل)",
]
_INSTRUCTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))
_PUNCT = re.compile(r"[^\w\s]+")


def looks_like_instruction(text: str) -> bool:
    """True when ``text`` addresses an AI assistant with instructions (en / fr / ar)."""
    folded = " ".join(_PUNCT.sub(" ", fold_for_search(text)).split())
    return bool(_INSTRUCTION.search(folded))
