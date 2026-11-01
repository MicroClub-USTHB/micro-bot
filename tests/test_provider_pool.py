"""Credential pools, key rotation and classified cool-downs."""

from __future__ import annotations

import json
from pathlib import Path

from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage, Provider
from micro_bot.llm.providers import (
    GROQ_BASE_URL,
    ProviderChain,
    _estimate_tokens,
    answer_providers,
    daily_calls,
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
    clients: dict[tuple[str, str], Client] = {}

    def make(provider: Provider) -> Client:
        client = Client(provider, [])
        clients[(provider.model, provider.key_id or provider.name)] = client
        return client

    chain = ProviderChain(
        configuration, answer_providers(configuration), clock=clock, client_factory=make
    )
    return chain, clients, clock


def keys(tmp_path: Path, **extra) -> list[str]:
    """One id per credential, in chain order: a fallback model repeats the same credentials."""
    return list(dict.fromkeys(p.key_id for p in answer_providers(settings(tmp_path, **extra))))


def test_a_pool_is_one_provider_with_one_slot_per_credential(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="a", gemini_api_keys="b,c")

    providers = [p for p in answer_providers(configuration) if p.model == "gemini-3.5-flash-lite"]

    assert [p.api_key for p in providers] == ["a", "b", "c"]
    assert len({p.key_id for p in providers}) == 3  # a quota file and a log field per credential
    # Both Gemini models are in the chain, so the pool count is per model and per credential
    assert pool_sizes(configuration) == {"gemini": 6}


def test_the_single_key_and_the_list_merge_without_duplicates(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_key="a", gemini_api_keys="a, b ,, b")

    assert list(dict.fromkeys(p.api_key for p in answer_providers(configuration))) == ["a", "b"]


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
    assert sorted(c.calls for c in clients.values() if c.calls) == [2, 2]  # the fallback sits idle


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

    picked = [chain._pick().provider.api_key for _ in range(4)]

    # round-robin would hand the 4th call to k2; fill-first drains k1's day first
    assert picked == ["k1", "k1", "k1", "k1"]


def test_a_spent_credential_moves_inside_the_same_provider_then_gives_up(tmp_path: Path) -> None:
    chain, _, _ = build(
        tmp_path,
        gemini_api_keys="k1,k2",
        llm_providers="gemini",
        gemini_daily_limit=1,
        gemini_rewrite_daily_limit=0,  # no fallback model here, so the chain can run out
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
    chain, _, clock = build(
        tmp_path, gemini_api_keys="k1", llm_providers="gemini", gemini_rewrite_daily_limit=0
    )

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
    chain, _, _ = build(
        tmp_path,
        groq_api_key="g",
        llm_providers="groq",
        groq_models="",
        groq_daily_tokens=100,
    )

    chain._settle(chain._pick(), GenerationResult("ok", GenerationUsage(1, 80, 20)))

    assert chain._slots[0].quota.tokens == 100
    assert chain._pick() is None  # 80 of 100 tokens: the binding limit is the TPD, not the RPD


def test_a_second_model_is_a_second_day_behind_the_first(tmp_path: Path) -> None:
    """Groq's free limits are a row per model, so 120b answers only once qwen's day is spent."""
    chain, _, _ = build(tmp_path, groq_api_key="g", llm_providers="groq", groq_daily_limit=1)

    picked = [chain._pick().provider.model for _ in range(2)]

    assert picked == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    assert chain._pick() is None  # both models' days are spent: providers_exhausted


def test_groq_is_configured_from_the_same_pool_helper(tmp_path: Path) -> None:
    configuration = settings(tmp_path, groq_api_key="g1,g2", llm_providers="groq")

    providers = answer_providers(configuration)

    # Model-major: every credential of qwen before any of 120b
    assert [(p.model, p.api_key) for p in providers] == [
        ("qwen/qwen3.8-27b", "g1"),
        ("qwen/qwen3.8-27b", "g2"),
        ("openai/gpt-oss-120b", "g1"),
        ("openai/gpt-oss-120b", "g2"),
    ]
    assert providers[0].base_url == GROQ_BASE_URL
    assert (providers[0].rpm, providers[0].tpm, providers[0].daily_limit) == (30, 8000, 1000)
    assert providers[0].daily_tokens == 200_000
    # qwen answers directly: at "low" it spent the output on thinking and returned nothing
    assert providers[0].extra == {"reasoning_effort": "none"}
    # gpt-oss bills its thinking as completion tokens, which Groq counts against the 200k TPD
    assert providers[2].extra == {"reasoning_effort": "low"}


def test_daily_calls_counts_every_pool_exactly_once(tmp_path: Path) -> None:
    configuration = settings(tmp_path, gemini_api_keys="a,b", groq_api_key="g")

    # Gemini counts per project and Groq per organization, so a second credential of the same one
    # adds nothing; two models are two rows in each provider's own table, so those do count twice
    assert daily_calls(configuration) == 2 * 500 + 2 * 500 + 1000 + 1000


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


def from_env(monkeypatch, tmp_path: Path, **variables: str) -> Settings:
    """A Settings built the way a deployment does it: from the environment, not kwargs."""
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "GROQ_API_KEY",
        "GROQ_API_KEYS",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEYS",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_four_gemini_keys_are_read_from_the_environment(monkeypatch, tmp_path: Path) -> None:
    configuration = from_env(
        monkeypatch,
        tmp_path,
        GEMINI_API_KEYS="AIza-one,AIza-two, AIza-three ,AIza-four",
    )

    providers = answer_providers(configuration)

    assert list(dict.fromkeys(p.api_key for p in providers)) == [
        "AIza-one",
        "AIza-two",
        "AIza-three",
        "AIza-four",
    ]
    assert len({p.key_id for p in providers}) == 4
    assert pool_sizes(configuration) == {"gemini": 8}  # four credentials at each of two models
    assert daily_calls(configuration) == 8 * 500


def test_blank_and_duplicate_entries_in_the_list_are_ignored(monkeypatch, tmp_path: Path) -> None:
    configuration = from_env(monkeypatch, tmp_path, GEMINI_API_KEYS="a,, a ,  ,b,")

    assert list(dict.fromkeys(p.api_key for p in answer_providers(configuration))) == ["a", "b"]


def test_the_singular_key_and_the_environment_list_are_merged(monkeypatch, tmp_path: Path) -> None:
    configuration = from_env(
        monkeypatch, tmp_path, GEMINI_API_KEY="first", GEMINI_API_KEYS="first,second"
    )

    assert list(dict.fromkeys(p.api_key for p in answer_providers(configuration))) == [
        "first",
        "second",
    ]


async def test_each_of_four_credentials_keeps_its_own_count_file(
    monkeypatch, tmp_path: Path
) -> None:
    chain, clients, _ = build_from_env(
        monkeypatch, tmp_path, GEMINI_API_KEYS="k1,k2,k3,k4", llm_providers="gemini"
    )

    for _ in range(4):
        await chain.generate("s", "u")

    counters = sorted((tmp_path / "llm_providers" / "gemini").glob("*/llm_quota.json"))
    assert len(counters) == 4
    assert all(json.loads(counter.read_text())["count"] == 1 for counter in counters)
    assert sorted(c.calls for c in clients.values() if c.calls) == [1, 1, 1, 1]


async def test_a_pool_of_four_serves_four_answers_then_reports_exhaustion(
    monkeypatch, tmp_path: Path
) -> None:
    chain, _, _ = build_from_env(
        monkeypatch,
        tmp_path,
        GEMINI_API_KEYS="k1,k2,k3,k4",
        llm_providers="gemini",
        GEMINI_DAILY_LIMIT="1",
        GEMINI_REWRITE_DAILY_LIMIT="0",  # only this pool is in play, so it can really run out
    )

    answers = [await chain.generate("s", "u") for _ in range(5)]

    assert [a.provider for a in answers[:4]] == ["gemini"] * 4
    assert answers[4].error == "providers_exhausted"


def test_cloudflare_accounts_are_read_from_the_environment(monkeypatch, tmp_path: Path) -> None:
    configuration = from_env(
        monkeypatch,
        tmp_path,
        CLOUDFLARE_ACCOUNTS="acc1:tok1,acc2:tok2",
    )

    cloudflare = [p for p in answer_providers(configuration) if p.name == "cloudflare"]

    assert [p.api_key for p in cloudflare] == ["tok1", "tok2"]
    assert "acc2" in cloudflare[1].base_url


def build_from_env(
    monkeypatch, tmp_path: Path, **variables: str
) -> tuple[ProviderChain, dict[str, Client], Clock]:
    """A chain over a Settings read from the environment, with one client per credential."""
    configuration = from_env(monkeypatch, tmp_path, **variables)
    clock = Clock()
    clients: dict[tuple[str, str], Client] = {}

    def make(provider: Provider) -> Client:
        client = Client(provider, [])
        clients[(provider.model, provider.key_id or provider.name)] = client
        return client

    chain = ProviderChain(
        configuration, answer_providers(configuration), clock=clock, client_factory=make
    )
    return chain, clients, clock


def test_tokens_are_recorded_only_for_a_provider_with_a_token_budget(tmp_path: Path) -> None:
    gemini, _, _ = build(tmp_path, gemini_api_key="g", llm_providers="gemini")
    gemini._settle(gemini._pick(), GenerationResult("ok", GenerationUsage(1, 900, 100)))
    counter = next((tmp_path / "llm_providers" / "gemini").glob("*/llm_quota.json"))

    assert "tokens" not in json.loads(counter.read_text())  # nothing reads it without a TPD


def test_the_token_cap_blocks_the_third_grounded_prompt_of_a_minute(tmp_path: Path) -> None:
    """Groq's 8000 TPM against a ~3.2k call is the limit a RAG prompt reaches, not the 30 RPM."""
    chain, _, clock = build(tmp_path, groq_api_key="g", llm_providers="groq", groq_models="")
    call = 3_200  # what one grounded prompt costs, as the guard estimates it

    picked = [chain._pick(call) for _ in range(3)]

    assert all(slot is not None for slot in picked[:2])  # 6400 of 8000 tokens fit
    assert picked[2] is None  # 9600 would be a 429 the guard exists to prevent
    clock.now += 61
    assert chain._pick(call) is not None  # the window is a minute, not a day


async def test_a_big_prompt_books_itself_against_the_minute(tmp_path: Path) -> None:
    """The per-minute guard reads the prompt before the call, so generate has to feed it."""
    chain, _, _ = build(
        tmp_path,
        groq_api_key="g",
        llm_providers="groq",
        groq_models="",
        groq_tpm=1000,
        llm_max_output_tokens=0,
    )
    prompt = "x" * 1_200  # three characters a token: 400 per call

    answers = [await chain.generate("", prompt) for _ in range(3)]

    assert [a.ok for a in answers[:2]] == [True, True]
    assert answers[2].error == "providers_exhausted"  # 1200 of 1000 tokens in one minute


def test_the_output_cap_is_part_of_the_estimate() -> None:
    """Thinking is billed as output, so a reasoning model's cap spends the minute's budget too."""
    assert _estimate_tokens("", "", 0) == 0
    assert _estimate_tokens("x" * 300, "x" * 300, 800) == 200 + 800


def test_a_daily_quota_429_cools_until_the_reset_not_for_centuries(tmp_path: Path) -> None:
    """The chain clock is monotonic; an epoch deadline would park the slot for 20,000 days."""
    chain, _, clock = build(
        tmp_path, gemini_api_key="k1", llm_providers="gemini", gemini_daily_limit=10
    )
    slot = chain._pick()

    chain._settle(slot, GenerationResult(None, GenerationUsage(1), "quota_exhausted"))

    pause = slot.blocked_until - clock.now
    assert 0 <= pause <= 86_400  # a day at most, and it comes back on its own


def test_the_global_cap_is_the_pools_once_not_once_per_call(tmp_path: Path) -> None:
    """Both calls draw on the same pools, so charging the answer pools and the rewrite pools apart
    would promise twice the requests the free tiers allow."""
    configuration = settings(tmp_path, gemini_api_key="g")

    assert daily_calls(configuration) == 1000  # the answer model's 500 + the rewrite model's 500
    assert daily_calls(configuration) != 2 * 1000
    assert daily_calls(settings(tmp_path, gemini_api_key="g", rewrite_enabled=False)) == 1000


def test_a_prompt_larger_than_the_minute_cap_is_still_attempted_once(tmp_path: Path) -> None:
    """Refusing it outright would silently drop the provider for every prompt of that size."""
    chain, _, _ = build(
        tmp_path, groq_api_key="g", llm_providers="groq", groq_models="", groq_tpm=1000
    )

    assert chain._pick(5_000) is not None  # over the cap, but the window is empty: try it
    assert chain._pick(5_000) is None  # never twice in the minute


def test_an_unknown_gemini_model_gets_no_reasoning_parameter(tmp_path: Path) -> None:
    """flash-lite rejects "none" and flash rejects "minimal": guessing is a 400 on every answer."""
    configuration = settings(tmp_path, gemini_api_key="g", gemini_model="gemini-4.0-flash-lite")

    assert answer_providers(configuration)[0].extra == {}


def test_qwen_is_asked_for_no_reasoning_budget(tmp_path: Path) -> None:
    """At "low" qwen spent 200 thinking tokens and returned empty content under the output cap."""
    configuration = settings(tmp_path, groq_api_key="g", llm_providers="groq")

    qwen = [p for p in answer_providers(configuration) if p.model.startswith("qwen/")]

    assert qwen and qwen[0].extra == {"reasoning_effort": "none"}


def test_each_chain_leads_with_its_own_model_and_trails_the_other(tmp_path: Path) -> None:
    """A spent pool on one call must not leave it without a provider while the other has quota."""
    configuration = settings(tmp_path, gemini_api_key="k", groq_api_key="g")

    answers = answer_providers(configuration)
    rewrites = rewrite_providers(configuration)

    assert answers[0].model == "gemini-3.5-flash-lite"
    assert rewrites[0].model == "gemini-3.1-flash-lite"
    assert answers[-1].model == "gemini-3.1-flash-lite"  # the rewrite model, as a last resort
    assert rewrites[-1].model == "gemini-3.5-flash-lite"  # the answer model, as a last resort
    assert [p.name for p in answers] == [p.name for p in rewrites]


async def test_a_spent_answer_pool_is_served_by_the_rewrite_model(tmp_path: Path) -> None:
    chain, _, _ = build(tmp_path, gemini_api_key="k", llm_providers="gemini", gemini_daily_limit=1)

    await chain.generate("s", "u")
    await chain.generate("s", "u")

    counters = sorted((tmp_path / "llm_providers" / "gemini").glob("*/llm_quota.json"))
    assert [counter.parent.name.split("__")[0] for counter in counters] == [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    assert [json.loads(counter.read_text())["count"] for counter in counters] == [1, 1]


def test_a_daily_429_with_a_hint_pauses_for_the_hint_not_the_midnight(tmp_path: Path) -> None:
    """Groq's TPD 429 says "6m 11.52s": parking that row until midnight loses the rest of a day."""
    chain, _, clock = build(tmp_path, gemini_api_key="k", llm_providers="gemini")
    pool = keys(tmp_path, gemini_api_key="k")
    slot = chain._pick()

    chain._settle(
        slot, GenerationResult(None, GenerationUsage(1), "quota_exhausted", retry_after_s=371.5)
    )

    assert 0 <= slot.blocked_until - clock.now <= 372  # six minutes, not the rest of the day
    counter = tmp_path / "llm_providers" / "gemini" / f"gemini-3.5-flash-lite__{pool[0]}"
    assert json.loads((counter / "llm_quota.json").read_text())["count"] == 1  # day not spent
    clock.now += 372
    assert chain._pick() is slot  # it comes back on its own hint


async def test_an_empty_reply_cools_the_row_and_hands_over(tmp_path: Path) -> None:
    """qwen returned empty content under a tight output cap: a lost answer is not a cheap one."""
    chain, clients, _ = build(tmp_path, groq_api_key="g", llm_providers="groq")
    qwen = next(client for (model, _), client in clients.items() if model.startswith("qwen/"))
    qwen.outcomes.append(GenerationResult(None, GenerationUsage(1), "invalid_response"))

    failed = await chain.generate("s", "u")
    handed_over = await chain.generate("s", "u")

    assert failed.error == "invalid_response"
    assert handed_over.ok  # the second row answers instead of the empty one being retried


def host_env(tmp_path: Path, **lines: str) -> Settings:
    """Settings read the way the host pastes them: one comma-separated line per provider."""
    path = tmp_path / "host.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in lines.items()), encoding="utf-8")
    return Settings(_env_file=path)


def chain_for(configuration: Settings, clock: Clock | None = None) -> tuple[ProviderChain, dict]:
    clients: dict[tuple[str, str], Client] = {}

    def make(provider: Provider) -> Client:
        client = Client(provider, [])
        clients[(provider.model, provider.key_id)] = client
        return client

    chain = ProviderChain(
        configuration, answer_providers(configuration), clock=clock or Clock(), client_factory=make
    )
    return chain, clients


async def test_a_pool_pasted_as_one_line_is_one_credential_per_entry(tmp_path: Path) -> None:
    """GEMINI_API_KEYS=k1,k2,k3,k4 with no spaces: four credentials, each with its own counter."""
    configuration = host_env(
        tmp_path,
        **{
            "DATA_DIR": str(tmp_path / "data"),
            "LLM_PROVIDERS": "gemini,groq",
            "GEMINI_API_KEYS": "g1,g2,g3,g4",
            "GROQ_API_KEYS": "q1,q2,q3,q4,q5",
        },
    )
    chain, clients = chain_for(configuration)

    await chain.generate("s", "u")
    await chain.generate("s", "u")
    await chain.generate("s", "u")
    await chain.generate("s", "u")

    assert pool_sizes(configuration) == {"gemini": 8, "groq": 10}
    # the answer model's four keys each served one call, so least-used rotated over the paste
    answer_calls = [
        c.calls for (model, _), c in clients.items() if model == "gemini-3.5-flash-lite"
    ]
    assert sorted(answer_calls) == [1, 1, 1, 1]
    counters = (tmp_path / "data" / "llm_providers" / "gemini").glob("*/llm_quota.json")
    assert len(list(counters)) == 4  # one file per pasted key, none shared


def test_rotation_strategies_read_the_pasted_pool(tmp_path: Path) -> None:
    def picked(strategy: str) -> list[str]:
        configuration = host_env(
            tmp_path,
            **{
                "DATA_DIR": str(tmp_path / strategy),
                "LLM_PROVIDERS": "gemini",
                "LLM_KEY_STRATEGY": strategy,
                "GEMINI_API_KEYS": "g1,g2,g3,g4",
            },
        )
        chain, _ = chain_for(configuration)
        return [chain._pick().provider.api_key for _ in range(4)]

    assert picked("round-robin") == ["g1", "g1", "g1", "g2"]  # sticky for three, then moves on
    assert picked("fill-first") == ["g1", "g1", "g1", "g1"]  # drains one key's day first


def test_a_pasted_pool_rotates_around_a_cooled_key_and_each_spent_one(tmp_path: Path) -> None:
    configuration = host_env(
        tmp_path,
        **{
            "DATA_DIR": str(tmp_path / "cool"),
            "LLM_PROVIDERS": "gemini",
            "GEMINI_API_KEYS": "g1,g2,g3,g4",
        },
    )
    chain, _ = chain_for(configuration)
    cooled = chain._pick()

    chain._settle(cooled, GenerationResult(None, GenerationUsage(1), "rate_limited", 45.0))

    handed_over = chain._pick()
    assert handed_over is not None  # one 429'd key does not take the pasted pool out
    assert handed_over.provider.key_id != cooled.provider.key_id

    spent = host_env(
        tmp_path,
        **{
            "DATA_DIR": str(tmp_path / "spent"),
            "LLM_PROVIDERS": "gemini",
            "GEMINI_API_KEYS": "g1,g2,g3,g4",
            "GEMINI_DAILY_LIMIT": "1",
            "GEMINI_REWRITE_DAILY_LIMIT": "0",
        },
    )
    chain, _ = chain_for(spent)
    keys_used = [chain._pick().provider.key_id for _ in range(4)]

    assert len(set(keys_used)) == 4  # four pasted keys served four questions
    assert chain._pick() is None  # and the pool is genuinely empty afterwards
