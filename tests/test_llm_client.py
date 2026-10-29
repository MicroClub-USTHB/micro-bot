from __future__ import annotations

from dataclasses import dataclass

from micro_bot.config import Settings
from micro_bot.llm.client import OPENROUTER_BASE_URL, OpenRouterClient


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
    def __init__(self, status_code: int, text: str = "", headers: dict | None = None) -> None:
        super().__init__()
        self.status_code = status_code
        self.response = type("Response", (), {"text": text, "headers": headers or {}})()


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


def test_reasoning_is_turned_off_unless_allowed() -> None:
    fake = FakeClient([FakeCompletion("a"), FakeCompletion("b")])
    OpenRouterClient(settings(), client=fake).generate("s", "u")
    on = settings().model_copy(update={"llm_disable_reasoning": False})
    OpenRouterClient(on, client=fake).generate("s", "u")
    first, second = fake.chat.completions.calls
    assert first["extra_body"] == {"reasoning": {"enabled": False}}
    assert "extra_body" not in second


def test_the_daily_quota_429_is_not_a_minute_429() -> None:
    body = (
        '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"@type":'
        '"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":'
        '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"@type":'
        '"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"33s"}]}}'
    )
    fake = FakeClient([ProviderError(429, body)])
    client = OpenRouterClient(settings(), client=fake)

    result = client.generate("system", "question")

    assert result.error == "quota_exhausted"  # the day is gone, not the minute
    assert result.retry_after_s == 33


def test_a_per_minute_429_keeps_the_provider_retry_after() -> None:
    fake = FakeClient([ProviderError(429, "quota exceeded", {"retry-after": "5m"})])

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    assert (result.error, result.retry_after_s) == ("rate_limited", 300)


def test_a_prose_retry_hint_is_read_in_seconds() -> None:
    fake = FakeClient(
        [ProviderError(429, "Rate limit reached for model. Please try again in 7.5s.")]
    )

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    assert (result.error, result.retry_after_s) == ("rate_limited", 7.5)


def test_a_groq_day_429_is_the_day_and_not_the_minute() -> None:
    body = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01` on tokens "
        "per day (TPD): Limit 200000, Used 199500, Requested 2500. Please try again in 1h2m3s."
    )
    fake = FakeClient([ProviderError(429, body)])

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    assert result.error == "quota_exhausted"  # the credential is spent until its own reset


def test_a_groq_minute_429_takes_the_token_reset_not_the_day_reset() -> None:
    body = (
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
        "Limit 8000, Used 7900, Requested 2500."
    )
    headers = {"x-ratelimit-reset-tokens": "7.66s", "x-ratelimit-reset-requests": "2m59.56s"}
    fake = FakeClient([ProviderError(429, body, headers)])

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    # the token cap is the binding one here, so its reset is the hint worth keeping
    assert (result.error, result.retry_after_s) == ("rate_limited", 7.66)


def test_a_per_minute_429_that_mentions_daily_usage_is_not_a_spent_day() -> None:
    """A bare "daily" once spent the whole credential; the word needs a limit next to it."""
    body = (
        "Rate limit reached for model `openai/gpt-oss-120b` on requests per minute (RPM): "
        "Limit 30, Used 30, Requested 1. See your daily usage at console.groq.com/settings/limits."
    )
    fake = FakeClient([ProviderError(429, body)])

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    assert result.error == "rate_limited"  # paused for the minute, not spent for the day


def test_a_daily_limit_in_prose_is_still_the_days_quota() -> None:
    fake = FakeClient([ProviderError(429, "You have exceeded your daily limit for this model.")])

    result = OpenRouterClient(settings(), client=fake).generate("system", "question")

    assert result.error == "quota_exhausted"


def test_the_answer_is_capped_in_tokens() -> None:
    fake = FakeClient([FakeCompletion("a"), FakeCompletion("b")])

    capped = settings().model_copy(update={"llm_max_output_tokens": 42})
    OpenRouterClient(capped, client=fake).generate("s", "u")
    uncapped = settings().model_copy(update={"llm_max_output_tokens": 0})
    OpenRouterClient(uncapped, client=fake).generate("s", "u")

    assert fake.chat.completions.calls[0]["max_tokens"] == 42  # thinking is billed as output
    assert "max_tokens" not in fake.chat.completions.calls[1]
