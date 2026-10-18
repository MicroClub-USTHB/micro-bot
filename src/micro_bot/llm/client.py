"""OpenRouter-only single-generation client with explicit failure results."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI

from micro_bot.config import Settings, get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class OpenAIClient(Protocol):
    chat: _Chat


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

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


ClientFactory = Callable[..., OpenAIClient]


class OpenRouterClient:
    """Make at most one OpenRouter generation per question."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenAIClient | None = None,
        client_factory: ClientFactory = OpenAI,
        async_client_factory: ClientFactory = AsyncOpenAI,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._client_factory = client_factory
        self._async_client_factory = async_client_factory
        self._async_client: Any = None

    def generate(self, system: str, user: str) -> GenerationResult:
        """Generate once, returning a result instead of provider exceptions."""
        if not self.settings.openrouter_api_key:
            return GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")

        client = self._client or self._client_factory(
            api_key=self.settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=self.settings.llm_timeout_s,
            max_retries=0,
        )
        try:
            completion = client.chat.completions.create(**self._request(system, user))
        except Exception as exc:  # Provider/SDK failures must not reach the API layer.
            return GenerationResult(None, GenerationUsage(llm_calls=1), self._error_kind(exc))

        return self._parse_completion(completion, calls=1)

    def _request(self, system: str, user: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "timeout": self.settings.llm_timeout_s,
        }
        if self.settings.llm_disable_reasoning:
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
        if not self.settings.openrouter_api_key:
            yield GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")
            return
        if self._async_client is None:
            self._async_client = self._async_client_factory(
                api_key=self.settings.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=self.settings.llm_timeout_s,
                max_retries=0,
            )
        parts: list[str] = []
        usage = None
        error: str | None = None
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
            error = self._error_kind(exc)
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
        status_code = _status_code(exc)
        if status_code == 429:
            return "rate_limited"
        if 500 <= status_code <= 599:
            return "provider_5xx"
        timeout_errors = (
            APITimeoutError,
            APIConnectionError,
            httpx.TimeoutException,
            TimeoutError,
        )
        if isinstance(exc, timeout_errors):
            return "timeout"
        return "provider_error"


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
