from __future__ import annotations

from dataclasses import dataclass

from mcclub_rag.config import Settings
from mcclub_rag.llm.client import OPENROUTER_BASE_URL, OpenRouterClient


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeCompletion:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def choices(self) -> list[FakeChoice]:
        return [FakeChoice(FakeMessage(self.content))]

    @property
    def usage(self) -> FakeUsage:
        return FakeUsage(self.prompt_tokens, self.completion_tokens)


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(outcomes)})()


def settings() -> Settings:
    return Settings(
        openrouter_api_key="test-key",
        openrouter_model="only/allowed-model",
        llm_timeout_s=17,
    )


def test_client_is_pinned_to_openrouter_disables_sdk_retries_and_uses_settings_model() -> None:
    created: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeClient:
        created.update(kwargs)
        return FakeClient([FakeCompletion("Answer [doc-1]", 11, 7)])

    client = OpenRouterClient(settings(), client_factory=factory)
    result = client.generate("system", "question")

    assert created == {
        "api_key": "test-key",
        "base_url": OPENROUTER_BASE_URL,
        "timeout": 17,
        "max_retries": 0,
    }
    assert result.ok is True
    assert result.text == "Answer [doc-1]"
    assert result.usage.llm_calls == 1
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7


def test_client_does_not_retry_a_provider_5xx() -> None:
    fake = FakeClient([ProviderError(503), FakeCompletion("Must not be used")])
    client = OpenRouterClient(settings(), client=fake)

    result = client.generate("system", "question")

    assert result.text is None
    assert result.error == "provider_5xx"
    assert result.usage.llm_calls == 1
    assert len(fake.chat.completions.calls) == 1


def test_client_does_not_retry_rate_limits_and_returns_a_degraded_result() -> None:
    fake = FakeClient([ProviderError(429)])
    client = OpenRouterClient(settings(), client=fake)

    result = client.generate("system", "question")

    assert result == result.__class__(None, result.usage, "rate_limited")
    assert result.usage.llm_calls == 1
    assert len(fake.chat.completions.calls) == 1


def test_client_timeout_returns_one_attempt_and_never_raises() -> None:
    fake = FakeClient([TimeoutError("timed out")])
    client = OpenRouterClient(settings(), client=fake)

    result = client.generate("system", "question")

    assert result.text is None
    assert result.error == "timeout"
    assert result.usage.llm_calls == 1
    assert len(fake.chat.completions.calls) == 1


def test_client_without_key_makes_no_request() -> None:
    fake = FakeClient([FakeCompletion("must not be consumed")])
    client = OpenRouterClient(Settings(openrouter_api_key=None), client=fake)

    result = client.generate("system", "question")

    assert result.text is None
    assert result.error == "not_configured"
    assert result.usage.llm_calls == 0
    assert fake.chat.completions.calls == []


def test_reasoning_is_turned_off_only_when_asked() -> None:
    fake = FakeClient([FakeCompletion("a"), FakeCompletion("b")])
    OpenRouterClient(settings(), client=fake).generate("s", "u")
    off = settings().model_copy(update={"llm_disable_reasoning": True})
    OpenRouterClient(off, client=fake).generate("s", "u")
    first, second = fake.chat.completions.calls
    assert "extra_body" not in first
    assert second["extra_body"] == {"reasoning": {"enabled": False}}
