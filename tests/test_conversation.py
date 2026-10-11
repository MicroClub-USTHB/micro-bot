from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from mcclub_rag.conversation import ConversationStore, Turn


class FakeClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 10, tzinfo=UTC))


def test_create_save_and_load_conversation(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)

    conversation = store.create("fr")
    saved = store.save(
        conversation.conversation_id,
        Turn(user_message="Quand est l'atelier ?", assistant_message="Samedi."),
        cited_chunk_ids=["chunk-1", "chunk-2"],
    )

    assert conversation.conversation_id.startswith("c_")
    UUID(conversation.conversation_id.removeprefix("c_"), version=4)
    assert saved is not None
    assert saved.language == "fr"
    assert saved.messages == [
        Turn(user_message="Quand est l'atelier ?", assistant_message="Samedi.")
    ]
    assert saved.chunk_ids == ["chunk-1", "chunk-2"]
    assert store.load(conversation.conversation_id) == saved


def test_save_keeps_only_the_last_three_turns_and_latest_citations(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)
    conversation = store.create("en")

    for number in range(4):
        store.save(
            conversation.conversation_id,
            Turn(user_message=f"question {number}", assistant_message=f"answer {number}"),
            cited_chunk_ids=[f"chunk-{number}"],
        )

    loaded = store.load(conversation.conversation_id)

    assert loaded is not None
    assert [turn.user_message for turn in loaded.messages] == [
        "question 1",
        "question 2",
        "question 3",
    ]
    assert loaded.chunk_ids == ["chunk-3"]


def test_load_expires_conversations_older_than_thirty_minutes(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)
    conversation = store.create("ar")

    clock.advance(timedelta(minutes=30))
    assert store.load(conversation.conversation_id) is not None

    clock.advance(timedelta(microseconds=1))
    assert store.load(conversation.conversation_id) is None
    assert (
        store.save(
            conversation.conversation_id,
            Turn(user_message="متى؟", assistant_message="السبت."),
            cited_chunk_ids=["chunk-1"],
        )
        is None
    )


def test_short_follow_up_reuses_the_pinned_language(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)
    conversation = store.create("ar")

    assert store.reuse_language(conversation.conversation_id, "ومتى؟") == "ar"
    assert (
        store.reuse_language(conversation.conversation_id, "When does the workshop begin?") is None
    )


def test_follow_up_with_exactly_twenty_letters_does_not_reuse_language(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)
    conversation = store.create("fr")

    assert store.reuse_language(conversation.conversation_id, "a" * 19) == "fr"
    assert store.reuse_language(conversation.conversation_id, "a" * 20) is None


def test_create_rejects_a_language_outside_query_lang(clock: FakeClock) -> None:
    store = ConversationStore(clock=clock)

    with pytest.raises(ValueError, match="QueryLang"):
        store.create("es")  # type: ignore[arg-type]
