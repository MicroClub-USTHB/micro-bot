"""Single-generation client for one OpenAI-compatible provider, with explicit failure results.

OpenRouter by default; ``Provider`` points it at another endpoint (Gemini, Cloudflare).
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI

from micro_bot.config import Settings, get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# A 429 whose body names a daily budget is the day's pool, not a minute's: the credential is
# spent until the provider's reset instead of paused for a few seconds.
_DAILY_QUOTA = re.compile(
    r"PerDay|per[-_ ]day|daily|free-models-per-day|\b(?:TPD|RPD)\b|tokens? per day"
    r"|requests? per day",
    re.IGNORECASE,
)
# "7.5s", "5m", "1h2m3s", "60" (Groq's Retry-After sends relative units)
_DURATION = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$", re.IGNORECASE
)
_TRY_AGAIN = re.compile(r"try again in\s+([\dhms.]+)", re.IGNORECASE)
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"([^"]+)"')


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class OpenAIClient(Protocol):
    chat: _Chat


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible endpoint and the model it answers with, for one credential."""

    name: str
    base_url: str
    api_key: str
    model: str
    daily_limit: int
    extra: dict[str, Any] = field(default_factory=dict)  # merged into every request
    rpm: int = 0  # requests per minute the free tier allows; 0 = no known limit
    # Short hash of the credential: names the quota file and the log field, never the key itself
    key_id: str = ""
    daily_tokens: int = 0  # per-day token cap (Groq's TPD); 0 = no known cap
    quota_reset_tz: str = "UTC"  # the zone its daily counter resets in (Gemini: Pacific)


@dataclass(frozen=True)
class GenerationUsage:
    """Generation outcome with a truthful count of attempted provider requests.

    A request that never starts has ``llm_calls=0``. A provider request,
    including a timeout, 429, or 5xx response, has ``llm_calls=1``.
    """

    llm_calls: int
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    text: str | None
    usage: GenerationUsage
    error: str | None = None
    provider: str | None = None  # which provider answered (logged, never sent to clients)
    retry_after_s: float | None = None  # the provider's own hint on a 429, when it gave one

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


ClientFactory = Callable[..., OpenAIClient]


class OpenRouterClient:
    """Make at most one generation per call, on OpenRouter or on ``provider``."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAIClient | None = None,
        client_factory: ClientFactory = OpenAI,
        async_client_factory: ClientFactory = AsyncOpenAI,
        provider: Provider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.api_key = provider.api_key if provider else self.settings.openrouter_api_key
        self.base_url = provider.base_url if provider else OPENROUTER_BASE_URL
        self.model = provider.model if provider else self.settings.openrouter_model
        self._client = client
        self._client_factory = client_factory
        self._async_client_factory = async_client_factory
        self._async_client: Any = None

    def generate(self, system: str, user: str) -> GenerationResult:
        """Generate once, returning a result instead of provider exceptions."""
        if not self.api_key:
            return GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")

        client = self._client or self._client_factory(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.settings.llm_timeout_s,
            max_retries=0,
        )
        try:
            completion = client.chat.completions.create(**self._request(system, user))
        except Exception as exc:  # Provider/SDK failures must not reach the API layer.
            error, retry_after_s = _failure(exc)
            return GenerationResult(
                None, GenerationUsage(llm_calls=1), error, retry_after_s=retry_after_s
            )

        return self._parse_completion(completion, calls=1)

    def _request(self, system: str, user: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": self.settings.llm_timeout_s,
        }
        if self.settings.llm_max_output_tokens:
            request["max_tokens"] = self.settings.llm_max_output_tokens
        if self.provider is not None:
            request.update(self.provider.extra)
        elif self.settings.llm_disable_reasoning:
            # Reasoning models (nemotron) spend 500-1000 hidden tokens and ~20 s per call otherwise.
            request["extra_body"] = {"reasoning": {"enabled": False}}
        return request

    async def stream(self, system: str, user: str) -> AsyncIterator[str | GenerationResult]:
        """One streamed generation: text deltas as they arrive, then the ``GenerationResult``.

        OpenRouter sends a 429 or 5xx as a plain HTTP error before the stream starts (the SDK
        raises), but an error after it started as a chunk with ``error`` /
        ``finish_reason="error"`` inside a 200, which the SDK does not raise: both become the
        result's ``error``. Usage comes in the last chunk. ``timeout`` is per read, so a
        stream that keeps sending is not cut. Closing the stream early stops the request, but
        the call still counts (Google, which serves the free gemma, can't cancel).
        """
        if not self.api_key:
            yield GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")
            return
        if self._async_client is None:
            self._async_client = self._async_client_factory(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.settings.llm_timeout_s,
                max_retries=0,
            )
        parts: list[str] = []
        usage = None
        error: str | None = None
        retry_after_s: float | None = None
        try:
            stream = await self._async_client.chat.completions.create(
                **self._request(system, user),
                stream=True,
                stream_options={"include_usage": True},
            )
            async with stream:
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None) or usage
                    if (getattr(chunk, "model_extra", None) or {}).get("error") is not None:
                        error = "provider_error"
                        break
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    if getattr(choices[0], "finish_reason", None) == "error":
                        error = "provider_error"
                        break
                    text = getattr(getattr(choices[0], "delta", None), "content", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
                        yield text
        except Exception as exc:  # Provider/SDK failures must not reach the API layer.
            error, retry_after_s = _failure(exc)
        content = "".join(parts)
        if error is None and not content.strip():
            error = "invalid_response"
        yield GenerationResult(
            None if error else content,
            GenerationUsage(
                llm_calls=1,
                prompt_tokens=_token_count(usage, "prompt_tokens"),
                completion_tokens=_token_count(usage, "completion_tokens"),
            ),
            error,
            retry_after_s=retry_after_s,
        )

    @staticmethod
    def _parse_completion(completion: Any, calls: int) -> GenerationResult:
        status_code = getattr(completion, "status_code", None)
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return GenerationResult(None, GenerationUsage(llm_calls=calls), "provider_5xx")
        if isinstance(status_code, int) and status_code == 429:
            return GenerationResult(None, GenerationUsage(llm_calls=calls), "rate_limited")
        if getattr(completion, "error", None) is not None:
            return GenerationResult(None, GenerationUsage(llm_calls=calls), "invalid_response")

        choices = getattr(completion, "choices", None)
        content = None
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            return GenerationResult(None, GenerationUsage(llm_calls=calls), "invalid_response")

        usage = getattr(completion, "usage", None)
        return GenerationResult(
            content,
            GenerationUsage(
                llm_calls=calls,
                prompt_tokens=_token_count(usage, "prompt_tokens"),
                completion_tokens=_token_count(usage, "completion_tokens"),
            ),
        )

    @staticmethod
    def _error_kind(exc: Exception) -> str:
        return _failure(exc)[0]


def _failure(exc: Exception) -> tuple[str, float | None]:
    """The error code and the provider's own retry hint, from an SDK or ``httpx`` exception.

    A 429 splits in two: a body that names a daily budget is ``quota_exhausted`` (the credential
    is done until the provider's reset), anything else is ``rate_limited`` (a minute at most).
    """
    status_code = _status_code(exc)
    if status_code == 429:
        body = _error_text(exc)
        kind = "quota_exhausted" if _DAILY_QUOTA.search(body) else "rate_limited"
        return kind, _retry_hint(exc, body)
    if 500 <= status_code <= 599:
        return "provider_5xx", None
    timeout_errors = (
        APITimeoutError,
        APIConnectionError,
        httpx.TimeoutException,
        TimeoutError,
    )
    if isinstance(exc, timeout_errors):
        return "timeout", None
    return "provider_error", None


def _error_text(exc: Exception) -> str:
    """The provider's message and body, which carry the quota id or the retry delay."""
    response = getattr(exc, "response", None)
    body = getattr(response, "text", "") if response is not None else ""
    return f"{exc} {body if isinstance(body, str) else ''}"


def _retry_hint(exc: Exception, body: str) -> float | None:
    """The provider's own cool-down: headers first, then the body."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    delay = _RETRY_DELAY.search(body)
    prose = _TRY_AGAIN.search(body)
    for hint in (
        headers.get("retry-after"),
        headers.get("x-ratelimit-reset-requests"),
        delay.group(1) if delay else None,
        prose.group(1).rstrip(".") if prose else None,
    ):
        seconds = _seconds(hint)
        if seconds is not None:
            return seconds
    return None


def _seconds(value: object) -> float | None:
    """Seconds from ``"7.5s"``, ``"5m"``, ``"1h2m"``, ``"60"``, an HTTP date or an epoch."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return _relative_or_epoch(float(value))
    text = str(value).strip()
    if not text:
        return None
    # Parse the unit before the number: int("5m") is 5, a 5 s cool-down instead of 5 minutes.
    duration = _DURATION.fullmatch(text)
    if duration is not None and any(duration.groups()):
        hours, minutes, seconds = (float(part or 0) for part in duration.groups())
        return hours * 3600 + minutes * 60 + seconds
    try:
        return _relative_or_epoch(float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _relative_or_epoch(value: float) -> float:
    """A small number is seconds; a big one is an epoch, in seconds (>1e9) or ms (>1e10)."""
    if value > 1e10:
        return max(0.0, value / 1000 - time.time())
    if value > 1e9:
        return max(0.0, value - time.time())
    return max(0.0, value)


def _status_code(exc: Exception) -> int:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else 0


def _token_count(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) and value >= 0 else 0
