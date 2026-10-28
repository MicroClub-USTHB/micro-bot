"""Credential pools, key rotation and classified cool-downs."""

from __future__ import annotations

import json
from pathlib import Path

from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage, Provider
from micro_bot.llm.providers import (
    GROQ_BASE_URL,
    ProviderChain,
    answer_providers,
    daily_capacity,
    pool_sizes,
    rewrite_providers,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Client:
    """One credential's client: replays its scripted outcomes and counts calls."""

    def __init__(self, provider: Provider, outcomes: list[object]) -> None:
        self.provider = provider
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, GenerationResult):
                return outcome
        return GenerationResult(f"from {self.provider.key_id}", GenerationUsage(1, 100, 20))

    async def stream(self, system: str, user: str):
        yield "Hi"
        yield GenerationResult("Hi", GenerationUsage(1))


def settings(tmp_path: Path, **extra) -> Settings:
    return Settings(data_dir=tmp_path, **extra)


def build(tmp_path: Path, **extra) -> tuple[ProviderChain, dict[str, Client], Clock]:
    """A chain over the configured pools, with one scripted client per credential."""
    configuration = settings(tmp_path, **extra)
    clock = Clock()
    clients: dict[str, Client] = {}

    def make(provider: Provider) -> Client:
        client = Client(provider, [])
        clients[provider.key_id or provider.name] = client
        return client

    chain = ProviderChain(
        configuration, answer_providers(configuration), clock=clock, client_factory=make
    )
    return chain, clients, clock


def keys(tmp_path: Path, **extra) -> list[str]:
    return [p.key_id for p in answer_providers(settings(tmp_path, **extra))]


def test_a_pool_is_one_provider_with_one_slot_per_credential(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="a", gemini_api_keys="b,c")

    providers = answer_providers(configuration)

    assert [p.name for p in providers] == ["gemini", "gemini", "gemini"]
    assert [p.api_key for p in providers] == ["a", "b", "c"]
    assert len({p.key_id for p in providers}) == 3  # a quota file and a log field per credential
    assert pool_sizes(configuration) == {"gemini": 3}


def test_the_single_key_and_the_list_merge_without_duplicates(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="a", gemini_api_keys="a, b ,, b")

    assert [p.api_key for p in answer_providers(configuration)] == ["a", "b"]


def test_cloudflare_pools_account_and_token_pairs(tmp_path: Path) -> None:
    configuration = settings(
        tmp_path,
        cloudflare_account_id="acc1",
        cloudflare_api_token="tok1",
        cloudflare_accounts="acc2:tok2",
    )

    cloudflare = [p for p in answer_providers(configuration) if p.name == "cloudflare"]

    assert len(cloudflare) == 2
    assert "acc1" in cloudflare[0].base_url and "acc2" in cloudflare[1].base_url
    assert [p.api_key for p in cloudflare] == ["tok1", "tok2"]


async def test_least_used_alternates_between_credentials(tmp_path: Path) -> None:
    chain, clients, _ = build(tmp_path, gemini_api_keys="k1,k2", llm_providers="gemini")
    pool = keys(tmp_path, gemini_api_keys="k1,k2")

    answers = [await chain.generate("s", "u") for _ in range(4)]

    assert [a.text for a in answers] == [f"from {key}" for key in pool * 2]
    assert sorted(client.calls for client in clients.values()) == [2, 2]


def test_round_robin_sticks_for_three_calls_then_moves_on(tmp_path: Path) -> None:
    chain, _, _ = build(
        tmp_path, gemini_api_keys="k1,k2", llm_providers="gemini", llm_key_strategy="round-robin"
    )
    pool = keys(tmp_path, gemini_api_keys="k1,k2")

    picked = [chain._pick().provider.key_id for _ in range(4)]

    assert picked == [pool[0], pool[0], pool[0], pool[1]]


def test_fill_first_drains_one_credential_before_the_next(tmp_path: Path) -> None:
    chain, _, _ = build(
        tmp_path, gemini_api_keys="k1,k2", llm_providers="gemini", llm_key_strategy="fill-first"
    )

    picked = [chain._pick().provider.api_key for _ in range(3)]

    assert picked == ["k1", "k1", "k1"]


def test_a_spent_credential_moves_inside_the_same_provider_then_gives_up(tmp_path: Path) -> None:
    chain, _, _ = build(
        tmp_path, gemini_api_keys="k1,k2", llm_providers="gemini", gemini_daily_limit=1
    )

    picked = [chain._pick().provider.api_key for _ in range(2)]

    assert picked == ["k1", "k2"]
    assert chain._pick() is None  # both credentials spent: providers_exhausted


def test_a_daily_quota_429_spends_the_credential_until_the_next_reset(tmp_path: Path) -> None:
    chain, _, _ = build(
        tmp_path, gemini_api_keys="k1,k2", llm_providers="gemini", gemini_daily_limit=10
    )
    pool = keys(tmp_path, gemini_api_keys="k1,k2")
    slot = chain._pick()

    chain._settle(slot, GenerationResult(None, GenerationUsage(1), "quota_exhausted"))

    counter = tmp_path / "llm_providers" / "gemini" / f"gemini-3.5-flash-lite__{pool[0]}"
    assert json.loads((counter / "llm_quota.json").read_text())["count"] == 10
    assert slot.blocked_until > chain.clock()  # out until the provider's own reset
    assert chain._pick().provider.key_id == pool[1]  # the other credential answers instead


def test_a_minute_429_cools_for_the_provider_retry_hint(tmp_path: Path) -> None:
    hint = GenerationResult(None, GenerationUsage(1), "rate_limited", retry_after_s=45)
    chain, _, clock = build(tmp_path, gemini_api_keys="k1", llm_providers="gemini")

    chain._settle(chain._pick(), hint)
    assert chain._pick() is None  # cooled for 45 s, and it is the only credential

    clock.now += 44
    assert chain._pick() is None
    clock.now += 2
    assert chain._pick() is not None


def test_repeated_failures_escalate_the_cooldown_and_a_success_clears_it(tmp_path: Path) -> None:
    chain, _, clock = build(tmp_path, gemini_api_key="k", llm_providers="gemini")
    slot = chain._slots[0]
    failure = GenerationResult(None, GenerationUsage(1), "provider_5xx")

    chain._settle(slot, failure)
    first = slot.blocked_until - clock.now
    chain._settle(slot, failure)
    second = slot.blocked_until - clock.now

    assert (first, second) == (120, 240)
    chain._settle(slot, GenerationResult("ok", GenerationUsage(1)))
    assert slot.level == 0


def test_tokens_are_recorded_and_a_token_budget_skips_the_provider(tmp_path: Path) -> None:
    chain, _, _ = build(tmp_path, groq_api_key="g", llm_providers="groq", groq_daily_tokens=100)

    chain._settle(chain._pick(), GenerationResult("ok", GenerationUsage(1, 80, 20)))

    assert chain._slots[0].quota.tokens == 100
    assert chain._pick() is None  # 80 of 100 tokens: the binding limit is the TPD, not the RPD


def test_groq_is_configured_from_the_same_pool_helper(tmp_path: Path) -> None:
    configuration = settings(tmp_path, groq_api_key="g1,g2", llm_providers="groq")

    providers = answer_providers(configuration)

    assert [p.api_key for p in providers] == ["g1", "g2"]
    assert providers[0].base_url == GROQ_BASE_URL
    assert (providers[0].model, providers[0].rpm, providers[0].daily_limit) == (
        "openai/gpt-oss-120b",
        30,
        1000,
    )
    assert providers[0].daily_tokens == 200_000


def test_daily_capacity_sums_every_pool(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_keys="a,b", groq_api_key="g")

    # Two Gemini credentials count twice, which is why the ToS note keeps them on one project
    assert daily_capacity(configuration) == 500 + 500 + 1000


def test_the_rewrite_uses_its_own_flash_lite_model_and_pool(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="g")

    rewrite = rewrite_providers(configuration)[0]

    assert (rewrite.model, rewrite.rpm, rewrite.daily_limit) == ("gemini-3.1-flash-lite", 15, 500)
    assert rewrite.extra == {"reasoning_effort": "minimal"}
    assert answer_providers(configuration)[0].model == "gemini-3.5-flash-lite"


def test_a_configured_reasoning_effort_overrides_the_model_table(tmp_path: Path) -> None:
    chain_settings = settings(tmp_path, gemini_api_key="g", gemini_reasoning_effort="none")

    assert answer_providers(chain_settings)[0].extra == {"reasoning_effort": "none"}


def test_the_model_table_picks_the_value_each_model_accepts(tmp_path: Path) -> None:
    flash = settings(tmp_path, gemini_api_key="g", gemini_model="gemini-3.8-flash")

    assert answer_providers(flash)[0].extra == {"reasoning_effort": "none"}


def test_reasoning_is_omitted_when_reasoning_is_allowed(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="g", gemini_model="gemini-3.8-flash")

    assert (
        answer_providers(configuration.model_copy(update={"llm_disable_reasoning": False}))[0].extra
        == {}
    )
