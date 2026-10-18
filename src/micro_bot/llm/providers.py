"""Free LLM providers tried in order, each with its own daily quota and failure cool-down.

No single free tier covers a day of club traffic, so the answer call goes to the first provider
(``LLM_PROVIDERS``) that has quota left today and has not failed recently. The provider is
picked *before* the call, never retried after a failure: a question still makes at most one
answer call (and one rewrite call). A provider that fails (429, 5xx, timeout) is skipped for
``LLM_PROVIDER_COOLDOWN_S``, so the next question goes to the next one. When none is left the
caller gets ``providers_exhausted`` and answers from the passages, as without a key.

The rewrite goes through ``ProviderChain.for_rewrite()``: the same order, but Gemini answers it
with its cheaper model and its own quota, so the answer model's quota pays only for answers.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import structlog

from micro_bot.config import Settings
from micro_bot.llm.client import (
    OPENROUTER_BASE_URL,
    GenerationResult,
    GenerationUsage,
    OpenRouterClient,
    Provider,
)
from micro_bot.llm.quota import DailyQuota

log = structlog.get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
_FAILURES = {"rate_limited", "provider_5xx", "timeout", "provider_error"}


def answer_providers(settings: Settings) -> list[Provider]:
    """Configured providers for the answer call, in ``LLM_PROVIDERS`` order."""
    return _ordered(settings, rewrite=False)


def rewrite_providers(settings: Settings) -> list[Provider]:
    """Same order; Gemini uses its rewrite model and quota."""
    return _ordered(settings, rewrite=True)


def _ordered(settings: Settings, *, rewrite: bool) -> list[Provider]:
    available = {p.name: p for p in _configured(settings, rewrite=rewrite)}
    names = [n.strip().lower() for n in settings.llm_providers.split(",") if n.strip()]
    return [available[n] for n in names if n in available]


def _configured(settings: Settings, *, rewrite: bool) -> list[Provider]:
    providers = []
    if settings.gemini_api_key:
        providers.append(
            Provider(
                name="gemini",
                base_url=GEMINI_BASE_URL,
                api_key=settings.gemini_api_key,
                model=settings.gemini_rewrite_model if rewrite else settings.gemini_model,
                daily_limit=(
                    settings.gemini_rewrite_daily_limit if rewrite else settings.gemini_daily_limit
                ),
                # Gemini models think by default: as little as each model allows
                extra=_gemini_extra(settings, rewrite=rewrite),
                rpm=settings.gemini_rewrite_rpm if rewrite else settings.gemini_rpm,
            )
        )
    if settings.cloudflare_account_id and settings.cloudflare_api_token:
        providers.append(
            Provider(
                name="cloudflare",
                base_url=CLOUDFLARE_BASE_URL.format(account_id=settings.cloudflare_account_id),
                api_key=settings.cloudflare_api_token,
                model=settings.cloudflare_model,
                daily_limit=settings.cloudflare_daily_limit,
                rpm=settings.cloudflare_rpm,
            )
        )
    if settings.openrouter_api_key:
        extra = {"extra_body": {"reasoning": {"enabled": False}}}
        providers.append(
            Provider(
                name="openrouter",
                base_url=OPENROUTER_BASE_URL,
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                daily_limit=settings.openrouter_daily_limit,
                rpm=settings.openrouter_rpm,
                extra=extra if settings.llm_disable_reasoning else {},
            )
        )
    return providers


def _gemini_extra(settings: Settings, *, rewrite: bool) -> dict:
    effort = (
        settings.gemini_rewrite_reasoning_effort if rewrite else settings.gemini_reasoning_effort
    )
    return {"reasoning_effort": effort} if effort and settings.llm_disable_reasoning else {}


@dataclass
class _Slot:
    provider: Provider
    client: OpenRouterClient
    quota: DailyQuota
    blocked_until: float = 0.0


class ProviderChain:
    """``GenerationService`` over several providers; see the module docstring."""

    def __init__(
        self,
        settings: Settings,
        providers: list[Provider],
        *,
        quota_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        client_factory: Callable[[Provider], OpenRouterClient] | None = None,
        _shared: tuple[Lock, dict[tuple[str, str], deque[float]]] | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self._root = quota_dir or settings.data_dir / "llm_providers"
        self._make = client_factory or (lambda p: OpenRouterClient(settings, provider=p))
        # One quota file per provider and model: the rewrite and answer chains share it when
        # they use the same model, and each Gemini model keeps its own.
        self._slots = [_Slot(p, self._make(p), self._quota(p)) for p in providers]
        # Calls of the last minute per provider and model, shared with the rewrite chain: the
        # per-minute caps (Gemini per model, OpenRouter per account) count both calls.
        self._lock, self._recent = _shared or (Lock(), {})
        self._rewrite: ProviderChain | None = None

    def _quota(self, provider: Provider) -> DailyQuota:
        path = self._root / provider.name / provider.model.replace("/", "_").replace("@", "")
        return DailyQuota(path, provider.daily_limit, 0)

    @property
    def names(self) -> list[str]:
        return [s.provider.name for s in self._slots]

    def for_rewrite(self) -> ProviderChain:
        """The chain the query rewrite uses (cheaper Gemini model, separate quota)."""
        if self._rewrite is None:
            self._rewrite = ProviderChain(
                self.settings,
                rewrite_providers(self.settings),
                quota_dir=self._root,
                clock=self.clock,
                client_factory=self._make,
                _shared=(self._lock, self._recent),
            )
        return self._rewrite

    def _pick(self) -> _Slot | None:
        now = self.clock()
        with self._lock:
            for slot in self._slots:
                if slot.blocked_until > now:
                    continue
                p = slot.provider
                recent = self._recent.setdefault((p.name, p.model), deque())
                while recent and recent[0] <= now - 60:
                    recent.popleft()
                if p.rpm and len(recent) >= p.rpm:
                    continue  # its per-minute cap would answer 429: the next provider instead
                if slot.quota.acquire():
                    recent.append(now)
                    return slot
        return None

    def _settle(self, slot: _Slot, result: GenerationResult) -> GenerationResult:
        if result.error in _FAILURES:
            s = self.settings
            pause = (
                s.llm_rate_limit_cooldown_s
                if result.error == "rate_limited"
                else s.llm_provider_cooldown_s
            )
            with self._lock:
                slot.blocked_until = self.clock() + pause
            log.warning("llm_provider_failed", provider=slot.provider.name, error=result.error)
        return replace(result, provider=slot.provider.name)

    async def generate(self, system: str, user: str) -> GenerationResult:
        slot = self._pick()
        if slot is None:
            return GenerationResult(None, GenerationUsage(llm_calls=0), "providers_exhausted")
        result = await asyncio.to_thread(slot.client.generate, system, user)
        return self._settle(slot, result)

    async def stream(self, system: str, user: str) -> AsyncIterator[str | GenerationResult]:
        slot = self._pick()
        if slot is None:
            yield GenerationResult(None, GenerationUsage(llm_calls=0), "providers_exhausted")
            return
        async for item in slot.client.stream(system, user):
            yield self._settle(slot, item) if isinstance(item, GenerationResult) else item
