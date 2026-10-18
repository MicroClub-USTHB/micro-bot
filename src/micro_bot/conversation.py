"""In-memory conversation context for chat follow-ups."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from micro_bot.schemas import QueryLang

_CONVERSATION_TTL = timedelta(minutes=30)
_MAX_TURNS = 3
_QUERY_LANGS = frozenset(("ar", "fr", "en"))


@dataclass(frozen=True)
class Turn:
    user_message: str
    assistant_message: str


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    messages: list[Turn]
    chunk_ids: list[str]
    language: QueryLang


@dataclass
class _StoredConversation:
    messages: list[Turn]
    chunk_ids: list[str]
    language: QueryLang
    expires_at: datetime


class ConversationStore:
    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._store: dict[str, _StoredConversation] = {}

    def create(self, language: QueryLang) -> Conversation:
        language = self._validate_language(language)
        conversation_id = f"c_{uuid4()}"
        self._store[conversation_id] = _StoredConversation(
            messages=[],
            chunk_ids=[],
            language=language,
            expires_at=self._now() + _CONVERSATION_TTL,
        )
        return self._snapshot(conversation_id, self._store[conversation_id])

    def load(self, conversation_id: str) -> Conversation | None:
        conversation = self._get_active(conversation_id)
        if conversation is None:
            return None
        return self._snapshot(conversation_id, conversation)

    def save(
        self,
        conversation_id: str,
        turn: Turn,
        cited_chunk_ids: Iterable[str],
    ) -> Conversation | None:
        conversation = self._get_active(conversation_id)
        if conversation is None:
            return None

        conversation.messages = [*conversation.messages, turn][-_MAX_TURNS:]
        conversation.chunk_ids = list(cited_chunk_ids)
        conversation.expires_at = self._now() + _CONVERSATION_TTL
        return self._snapshot(conversation_id, conversation)

    def reuse_language(self, conversation_id: str, message: str) -> QueryLang | None:
        conversation = self._get_active(conversation_id)
        if conversation is None or self._letter_count(message) >= 20:
            return None
        return conversation.language

    def _get_active(self, conversation_id: str) -> _StoredConversation | None:
        conversation = self._store.get(conversation_id)
        if conversation is None:
            return None
        if conversation.expires_at < self._now():
            del self._store[conversation_id]
            return None
        return conversation

    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _letter_count(message: str) -> int:
        return sum(character.isalpha() for character in message)

    @staticmethod
    def _validate_language(language: QueryLang) -> QueryLang:
        if language not in _QUERY_LANGS:
            raise ValueError("language must be a QueryLang: ar, fr, or en")
        return cast(QueryLang, language)

    @staticmethod
    def _snapshot(conversation_id: str, conversation: _StoredConversation) -> Conversation:
        return Conversation(
            conversation_id=conversation_id,
            messages=list(conversation.messages),
            chunk_ids=list(conversation.chunk_ids),
            language=conversation.language,
        )
