"""Fixed notices for the replies no model writes, in the question's language (I5).

Only the paths without a generation use them: an empty message, nothing retrieved, the LLM
unavailable (no key, budget spent, provider error), or a bare ``NOT_FOUND`` from the model.
Every other reply, refusals included, is written by the model in the user's language.
"""

NO_ANSWER: dict[str, str] = {
    "en": "I don't know based on the available documents.",
    "fr": "Je ne sais pas d'après les documents disponibles.",
    "ar": "لا أعرف بناءً على الوثائق المتاحة.",
}
UNAVAILABLE: dict[str, str] = {
    "en": "I can't write an answer right now. The most relevant passages are listed in the "
    "sources.",
    "fr": "Je ne peux pas rédiger de réponse pour le moment. Les passages les plus pertinents "
    "sont listés dans les sources.",
    "ar": "لا يمكنني كتابة إجابة الآن. أهم المقاطع ذات الصلة مذكورة في المصادر.",
}
EMPTY_MESSAGE: dict[str, str] = {
    "en": "Please type a question.",
    "fr": "Veuillez écrire une question.",
    "ar": "من فضلك اكتب سؤالا.",
}
