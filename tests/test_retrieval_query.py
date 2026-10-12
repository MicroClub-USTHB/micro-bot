import pytest

from mcclub_rag.conversation import Turn
from mcclub_rag.retrieval.query import build_search_query, has_followup_cue, normalize_query


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


@pytest.mark.parametrize(
    "message",
    [
        # leading conjunctions
        "And where?",
        "and who issues them",
        "What about the IT theme?",
        "how about members?",
        "Et pour les masters ?",
        "Alors combien ?",
        "وكم عدد النازحين؟",
        "وأين يقع المطار؟",
        "وشكون لي يعطيهم؟",
        "ثم ماذا حدث؟",
        # pronouns and demonstratives
        "Can they be appealed?",
        "Which database does it use?",
        "Who leads this?",
        "et son parcours au club ?",
        "Et elle a combien d'utilisateurs ?",
        "Quand a-t-il lieu ?",
        "C'est quand, ça ?",
        "متى يبدأ ذلك؟",
        "ماذا قال عنها؟",
        "شحال تاعو؟",
        "chhal dyalo ?",
        # ellipsis
        "...and the date?",
        "Where?",
        "وين؟",
    ],
)
def test_follow_up_cues(message):
    assert has_followup_cue(message)


@pytest.mark.parametrize(
    "message",
    [
        "Who is the dean?",
        "Qui est le doyen ?",
        "من هو العميد؟",
        "What time is the Python workshop?",
        "Qu'est-ce que le MCDI ?",
        "Y a-t-il une bibliothèque ?",
        "Is there a library?",
        "متى ورشة الأمن السيبراني؟",
        "وقت الامتحان؟",
        "مكتبة الكلية",
        "En quelle année ?",
    ],
)
def test_standalone_questions_have_no_cue(message):
    assert not has_followup_cue(message)


def test_short_question_without_cue_is_searched_alone():
    history = [turn("When is the MC Hackathon 2026?")]
    assert build_search_query("Who is the dean?", history, 8) == ("Who is the dean?", False)


def test_cue_rule_can_be_switched_off():
    history = [turn("When is the MC Hackathon 2026?")]
    query, used = build_search_query("Who is the dean?", history, 8, cues=False)
    assert used and query.endswith("Who is the dean?")


def test_normalize_query_collapses_repeated_punctuation_always():
    assert normalize_query("Quand ??!!  est   l'élection...", False, False) == (
        "Quand ? est l'élection..."
    )


def test_normalize_query_lowers_shouting_only():
    assert normalize_query("WHEN IS THE HACKATHON?", True, False) == "when is the hackathon?"
    assert normalize_query("Where is the USTHB library?", True, False) == (
        "Where is the USTHB library?"
    )
    assert normalize_query("WHEN IS THE HACKATHON?", False, False) == "WHEN IS THE HACKATHON?"
    assert normalize_query("MCDI", True, False) == "MCDI"  # too short to call it shouting


def test_normalize_query_strips_harakat_when_asked():
    assert normalize_query("أَيْنَ المَكْتَبَةُ؟", False, True) == "أين المكتبة؟"
    assert normalize_query("أَيْنَ", False, False) == "أَيْنَ"


def test_fix_case_repairs_scrambled_words_only():
    assert normalize_query("WhEn is the pYthon wOrkshop ??", True, False) == (
        "when is the python workshop ?"
    )
    assert normalize_query("Is USTHB on GitHub or McDonald?", True, False) == (
        "Is USTHB on GitHub or McDonald?"
    )
