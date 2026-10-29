"""Free LLM providers tried in order, each credential with its own daily quota and cool-down.

No single free tier covers a day of club traffic, so the answer call goes to the first provider
(``LLM_PROVIDERS``) that still has an eligible credential: quota left today, room under its
per-minute cap, and no recent failure. The provider is picked *before* the call, never retried
after a failure: a question still makes at most one answer call (and one rewrite call). A
credential that fails is cooled down or, on a daily-quota 429, taken out until the provider's own
reset, so the next question goes to the next one. When none is left the caller gets
``providers_exhausted`` and answers from the passages, as without a key.

A provider may hold several credentials (``*_API_KEYS``): each is its own pool,
its own per-minute window and its own cool-down, and ``LLM_KEY_STRATEGY`` decides which one
serves the next call. That is how the free tiers are actually used up rather than 429'd away.
A provider may also hold several models (``GROQ_MODELS``): Groq's free limits are a row per
model, so a second model is a second day of quota behind the first.

The per-minute guard counts tokens as well as requests, because a grounded prompt is ~2.5k
tokens and the token cap is the one a RAG assistant reaches first (Groq: 8k TPM against 30 RPM).

The rewrite goes through ``ProviderChain.for_rewrite()``: the same order, but Gemini answers it
with a cheaper model on its own quota, so the answer model's quota pays only for answers.
"""

from __future__ import annotations

import asyncio
import hashlib
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
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_FAILURES = {"rate_limited", "quota_exhausted", "provider_5xx", "timeout", "provider_error"}
_STICKY_USES = 3  # round-robin keeps one credential for this many calls, then moves on

# The lowest reasoning value each Gemini model accepts; Google's 3.x models cannot stop thinking
# and the wrong string is a 400 (the flash-lite models reject "none", the flash models reject
# "minimal"; "none" maps to thinking_level minimal anyway).
_GEMINI_EFFORT = {
    "gemini-3.5-flash-lite": "minimal",
    "gemini-3.1-flash-lite": "minimal",
    "gemini-3.6-flash": "none",
    "gemini-3.8-flash": "none",
}

# gpt-oss bills its thinking as completion tokens, and Groq's TPD counts them, so a grounded
# answer must spend as little on thinking as the model allows ("low" is gpt-oss's floor).
# A model missing here gets no parameter at all: an unsupported value is a 400, not a hint.
_GROQ_EFFORT = {
    "openai/gpt-oss-120b": "low",
    "openai/gpt-oss-20b": "low",
    "openai/gpt-oss-safeguard-20b": "low",
}


def answer_providers(settings: Settings) -> list[Provider]:
    """Configured credentials for the answer call, in ``LLM_PROVIDERS`` order."""
    return _ordered(settings, rewrite=False)


def rewrite_providers(settings: Settings) -> list[Provider]:
    """Same order; Gemini uses its rewrite model and quota."""
    return _ordered(settings, rewrite=True)


def daily_capacity(settings: Settings) -> int:
    """Total answer capacity of the configured pools (``LLM_DAILY_LIMIT=0`` means this)."""
    return sum(p.daily_limit for p in answer_providers(settings))


def pool_sizes(settings: Settings) -> dict[str, int]:
    """Slots per provider for the startup log: models times credentials on Groq."""
    sizes: dict[str, int] = {}
    for provider in answer_providers(settings):
        sizes[provider.name] = sizes.get(provider.name, 0) + 1
    return sizes


def _ordered(settings: Settings, *, rewrite: bool) -> list[Provider]:
    """Every credential of a provider, grouped in ``LLM_PROVIDERS`` order."""
    configured = _configured(settings, rewrite=rewrite)
    names = [n.strip().lower() for n in settings.llm_providers.split(",") if n.strip()]
    return [p for name in names for p in configured if p.name == name]


def _configured(settings: Settings, *, rewrite: bool) -> list[Provider]:
    providers: list[Provider] = []
    effort = _gemini_effort(settings, rewrite=rewrite)
    for key in _keys(settings.gemini_api_key, settings.gemini_api_keys):
        providers.append(
            Provider(
                name="gemini",
                base_url=GEMINI_BASE_URL,
                api_key=key,
                model=settings.gemini_rewrite_model if rewrite else settings.gemini_model,
                daily_limit=(
                    settings.gemini_rewrite_daily_limit if rewrite else settings.gemini_daily_limit
                ),
                # Gemini models think by default: as little as each model allows
                extra={"reasoning_effort": effort}
                if effort and settings.llm_disable_reasoning
                else {},
                rpm=settings.gemini_rewrite_rpm if rewrite else settings.gemini_rpm,
                key_id=_key_id(key),
                quota_reset_tz=settings.gemini_quota_reset_tz,
            )
        )
    for model in _keys(settings.groq_model, settings.groq_models):
        effort = settings.groq_reasoning_effort.strip() or _GROQ_EFFORT.get(model, "")
        for key in _keys(settings.groq_api_key, settings.groq_api_keys):
            providers.append(
                Provider(
                    name="groq",
                    base_url=GROQ_BASE_URL,
                    api_key=key,
                    model=model,
                    daily_limit=settings.groq_daily_limit,
                    rpm=settings.groq_rpm,
                    tpm=settings.groq_tpm,
                    key_id=_key_id(key),
                    daily_tokens=settings.groq_daily_tokens,
                    extra={"reasoning_effort": effort}
                    if effort and settings.llm_disable_reasoning
                    else {},
                )
            )
    for account_id, token in _cloudflare_accounts(settings):
        providers.append(
            Provider(
                name="cloudflare",
                base_url=CLOUDFLARE_BASE_URL.format(account_id=account_id),
                api_key=token,
                model=settings.cloudflare_model,
                daily_limit=settings.cloudflare_daily_limit,
                rpm=settings.cloudflare_rpm,
                key_id=_key_id(token),
            )
        )
    for key in _keys(settings.openrouter_api_key, settings.openrouter_api_keys):
        extra = {"extra_body": {"reasoning": {"enabled": False}}}
        providers.append(
            Provider(
                name="openrouter",
                base_url=OPENROUTER_BASE_URL,
                api_key=key,
                model=settings.openrouter_model,
                daily_limit=settings.openrouter_daily_limit,
                rpm=settings.openrouter_rpm,
                extra=extra if settings.llm_disable_reasoning else {},
                key_id=_key_id(key),
            )
        )
    return providers


def _gemini_effort(settings: Settings, *, rewrite: bool) -> str:
    """The configured value, or the model's own lowest accepted one (R5)."""
    model = settings.gemini_rewrite_model if rewrite else settings.gemini_model
    configured = (
        settings.gemini_rewrite_reasoning_effort if rewrite else settings.gemini_reasoning_effort
    ).strip()
    return configured or _GEMINI_EFFORT.get(model, "minimal")


def _keys(*values: str | None) -> list[str]:
    """A credential pool from the singular setting and its comma-separated list, deduplicated."""
    keys: list[str] = []
    for value in values:
        for part in (value or "").split(","):
            key = part.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def _cloudflare_accounts(settings: Settings) -> list[tuple[str, str]]:
    """``account_id:token`` pairs: the free neurons are per account, so each is its own pool."""
    pairs: list[tuple[str, str]] = []
    single = (settings.cloudflare_account_id or "", settings.cloudflare_api_token or "")
    if single[0].strip() and single[1].strip():
        pairs.append((single[0].strip(), single[1].strip()))
    for entry in settings.cloudflare_accounts.split(","):
        account_id, _, token = entry.strip().partition(":")
        pair = (account_id.strip(), token.strip())
        if pair[0] and pair[1] and pair not in pairs:
            pairs.append(pair)
    return pairs


def _key_id(key: str) -> str:
    """A short stable id for one credential: names its quota file and its log field only."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@dataclass
class _Slot:
    provider: Provider
    client: OpenRouterClient
    quota: DailyQuota
    blocked_until: float = 0.0
    level: int = 0  # consecutive failures, drives the escalating cool-down
    last_used: float = 0.0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider.name, self.provider.model, self.provider.key_id)


class ProviderChain:
    """``GenerationService`` over several providers and credentials; see the module docstring."""

    def __init__(
        self,
        settings: Settings,
        providers: list[Provider],
        *,
        quota_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        client_factory: Callable[[Provider], OpenRouterClient] | None = None,
        _shared: tuple[Lock, dict[tuple[str, str, str], deque[tuple[float, int]]]] | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.strategy = settings.llm_key_strategy
        self._root = quota_dir or settings.data_dir / "llm_providers"
        self._make = client_factory or (lambda p: OpenRouterClient(settings, provider=p))
        # One quota file per provider, model and credential: the rewrite and answer chains
        # share it when they use the same model, and each Gemini model keeps its own.
        self._slots = [_Slot(p, self._make(p), self._quota(p)) for p in providers]
        # Calls of the last minute per credential, as (moment, tokens), shared with the rewrite
        # chain: the per-minute caps (Gemini per model, Groq per key and model) count both calls.
        self._lock, self._recent = _shared or (Lock(), {})
        self._sticky: _Slot | None = None
        self._sticky_uses = 0
        self._rewrite: ProviderChain | None = None

    def _quota(self, provider: Provider) -> DailyQuota:
        model = provider.model.replace("/", "_").replace("@", "")
        suffix = f"__{provider.key_id}" if provider.key_id else ""
        return DailyQuota(
            self._root / provider.name / f"{model}{suffix}",
            provider.daily_limit,
            0,
            reset_tz=provider.quota_reset_tz,
            token_limit=provider.daily_tokens,
        )

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

    def _pick(self, cost: int = 0) -> _Slot | None:
        """The next eligible credential: provider order first, strategy inside a provider."""
        now = self.clock()
        with self._lock:
            for group in self._groups():
                candidates = [
                    s for s in self._slots if (s.provider.name, s.provider.model) == group
                ]
                for slot in self._candidates(candidates):
                    if not self._available(slot, now, cost):
                        continue
                    if slot.quota.acquire():
                        slot.last_used = now
                        self._recent[slot.key].append((now, cost))
                        self._remember(slot)
                        return slot
        return None

    def _groups(self) -> list[tuple[str, str]]:
        """``(provider, model)`` pairs in slot order, each once.

        Groq's free limits are a row per model, so a second model id is a second pool: grouping
        by model keeps the better one serving until its day is spent, and ``LLM_KEY_STRATEGY``
        rotates credentials *inside* a model instead of spreading a question onto a weaker model
        for the sake of levelling the call counts.
        """
        groups: list[tuple[str, str]] = []
        for slot in self._slots:
            group = (slot.provider.name, slot.provider.model)
            if group not in groups:
                groups.append(group)
        return groups

    def _candidates(self, candidates: list[_Slot]) -> list[_Slot]:
        """Order one provider's credentials by ``LLM_KEY_STRATEGY``."""
        if self.strategy == "fill-first":
            return candidates
        if self.strategy == "round-robin":
            sticky = self._sticky
            if (
                sticky is not None
                and self._sticky_uses < _STICKY_USES
                and any(s is sticky for s in candidates)
            ):
                return [sticky, *(s for s in candidates if s is not sticky)]
            return sorted(candidates, key=lambda s: (s.last_used, s.quota.used))
        # least-used: the credential with the fewest calls today, then the oldest one
        return sorted(candidates, key=lambda s: (s.quota.used, s.last_used))

    def _remember(self, slot: _Slot) -> None:
        if self._sticky is slot:
            self._sticky_uses += 1
        else:
            self._sticky, self._sticky_uses = slot, 1

    def _available(self, slot: _Slot, now: float, cost: int = 0) -> bool:
        """Not cooled down, not rate-limited this minute, not out of quota or tokens."""
        if slot.blocked_until > now:
            return False
        recent = self._recent.setdefault(slot.key, deque())
        while recent and recent[0][0] <= now - 60:
            recent.popleft()
        rpm = slot.provider.rpm
        if rpm and len(recent) >= rpm:
            return False  # its per-minute cap would answer 429: try another credential
        tpm = slot.provider.tpm
        if tpm and sum(tokens for _, tokens in recent) + cost > tpm:
            return False  # the token cap, which a grounded prompt hits long before the request cap
        return not (slot.quota.remaining == 0 or slot.quota.tokens_spent)

    def _settle(self, slot: _Slot, result: GenerationResult) -> GenerationResult:
        """Turn a failure into the right cool-down, and a success into a token record."""
        now = self.clock()
        if result.error == "quota_exhausted":
            with self._lock:
                slot.quota.exhaust()
                # The chain clock is monotonic, so the wait must be a duration, not an epoch:
                # parking the slot at next_reset() would block it for centuries.
                slot.blocked_until = now + slot.quota.seconds_to_reset()
                slot.level += 1
        elif result.error == "rate_limited":
            pause = result.retry_after_s or self.settings.llm_rate_limit_cooldown_s
            with self._lock:
                slot.blocked_until = now + pause
                slot.level += 1
        elif result.error in _FAILURES:
            pause = min(
                self.settings.llm_provider_cooldown_s * 2**slot.level,
                self.settings.llm_backoff_max_s,
            )
            with self._lock:
                slot.blocked_until = now + pause
                slot.level += 1
        else:
            with self._lock:
                slot.level = 0
            tokens = result.usage.prompt_tokens + result.usage.completion_tokens
            if tokens and slot.provider.daily_tokens:
                # Only a provider with a token budget reads this back (Groq's TPD)
                slot.quota.record_tokens(tokens)
        if result.error in _FAILURES:
            log.warning(
                "llm_provider_failed",
                provider=slot.provider.name,
                key=slot.provider.key_id,
                model=slot.provider.model,
                error=result.error,
                retry_after_s=result.retry_after_s,
                cooldown_s=max(0.0, slot.blocked_until - now),
            )
        return replace(result, provider=slot.provider.name)

    async def generate(self, system: str, user: str) -> GenerationResult:
        slot = self._pick(_estimate_tokens(system, user, self.settings.llm_max_output_tokens))
        if slot is None:
            return GenerationResult(None, GenerationUsage(llm_calls=0), "providers_exhausted")
        result = await asyncio.to_thread(slot.client.generate, system, user)
        return self._settle(slot, result)

    async def stream(self, system: str, user: str) -> AsyncIterator[str | GenerationResult]:
        slot = self._pick(_estimate_tokens(system, user, self.settings.llm_max_output_tokens))
        if slot is None:
            yield GenerationResult(None, GenerationUsage(llm_calls=0), "providers_exhausted")
            return
        async for item in slot.client.stream(system, user):
            yield self._settle(slot, item) if isinstance(item, GenerationResult) else item


def _estimate_tokens(system: str, user: str, max_output: int) -> int:
    """An upper bound for one call's tokens, which only the per-minute guard reads.

    Three characters per token, not four: Arabic and French tokenize worse than English, and an
    under-estimate would earn the 429 the guard exists to prevent. The output cap is added
    because a reasoning model spends it out of the same budget.
    """
    return (len(system) + len(user)) // 3 + max_output
