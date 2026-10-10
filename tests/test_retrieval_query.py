from mcclub_rag.conversation import Turn
from mcclub_rag.retrieval.query import build_search_query


def turn(user: str, assistant: str = "answer") -> Turn:
    return Turn(user_message=user, assistant_message=assistant)


def test_no_history_uses_the_message():
    assert build_search_query("Quand a lieu l'élection ?", [], 8) == (
        "Quand a lieu l'élection ?",
        False,
    )


def test_short_follow_up_is_joined_to_the_previous_question():
    history = [turn("Qui est candidat à la vice-présidence ?")]
    assert build_search_query("et son programme ?", history, 8) == (
        "Qui est candidat à la vice-présidence ?\net son programme ?",
        True,
    )


def test_long_message_stands_alone():
    history = [turn("Qui est candidat ?")]
    message = "Quelles sont les priorités du candidat pour les ateliers techniques du club ?"
    assert build_search_query(message, history, 8) == (message, False)


def test_only_the_last_turn_is_used():
    history = [turn("première question"), turn("deuxième question")]
    assert build_search_query("et ensuite ?", history, 8)[0] == "deuxième question\net ensuite ?"


def test_items_that_are_not_turns_are_ignored():
    assert build_search_query("et ensuite ?", ["not a turn", {"user_message": "x"}], 8) == (
        "et ensuite ?",
        False,
    )


def test_message_is_normalised():
    composed = "été  \t et​  ensuite"
    query, used = build_search_query(composed, [], 8)
    assert query == "été et ensuite" and not used


def test_empty_message_gives_an_empty_query():
    assert build_search_query(" \n\t ", [turn("question")], 8) == ("", False)


def test_zero_max_words_disables_history():
    assert build_search_query("et ensuite ?", [turn("question")], 0) == ("et ensuite ?", False)
