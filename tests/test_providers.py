"""The free-provider chain: order, per-provider quota, cool-down, one call per question."""

from pathlib import Path

from micro_bot.config import Settings
from micro_bot.llm.client import GenerationResult, GenerationUsage, Provider
from micro_bot.llm.providers import (
    CLOUDFLARE_BASE_URL,
    GEMINI_BASE_URL,
    ProviderChain,
    answer_providers,
    rewrite_providers,
)


def settings(tmp_path: Path, **extra) -> Settings:
    base = {
        "data_dir": tmp_path,
        "gemini_api_key": "g",
        "cloudflare_account_id": "acc",
        "cloudflare_api_token": "cf",
        "openrouter_api_key": "or",
    }
    return Settings(**(base | extra))


class FakeClient:
    def __init__(self, provider: Provider, results: dict[str, list[str | None]]) -> None:
        self.provider = provider
        self.results = results
        self.calls = 0

    def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        error = (
            self.results.get(self.provider.name, [None]).pop(0)
            if self.results.get(self.provider.name)
            else None
        )
        if error:
            return GenerationResult(None, GenerationUsage(1), error)
        return GenerationResult(f"from {self.provider.model}", GenerationUsage(1, 2, 3))

    async def stream(self, system: str, user: str):
        yield "Hi "
        yield GenerationResult("Hi ", GenerationUsage(1), None)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def chain(tmp_path: Path, s: Settings, results=None, clock=None) -> tuple[ProviderChain, dict]:
    clients: dict[str, FakeClient] = {}

    def make(p: Provider) -> FakeClient:
        clients[f"{p.name}:{p.model}"] = FakeClient(p, results if results is not None else {})
        return clients[f"{p.name}:{p.model}"]

    return ProviderChain(
        s, answer_providers(s), clock=clock or Clock(), client_factory=make
    ), clients


def test_providers_follow_the_configured_order_and_skip_missing_keys(tmp_path: Path) -> None:
    s = settings(tmp_path, llm_providers="openrouter, gemini,cloudflare", cloudflare_api_token=None)

    providers = answer_providers(s)

    assert [p.name for p in providers] == ["openrouter", "gemini"]
    gemini = providers[1]
    assert gemini.base_url == GEMINI_BASE_URL
    assert gemini.model == "gemini-3.5-flash-lite"
    assert gemini.extra == {"reasoning_effort": "minimal"}


def test_rewrite_uses_the_cheaper_gemini_model_and_its_own_limit(tmp_path: Path) -> None:
    s = settings(tmp_path, gemini_rewrite_daily_limit=7)

    gemini = rewrite_providers(s)[0]

    assert gemini.model == "gemini-3.5-flash-lite" and gemini.daily_limit == 7
    assert gemini.extra == {"reasoning_effort": "minimal"}


def test_cloudflare_url_carries_the_account_id(tmp_path: Path) -> None:
    cloudflare = answer_providers(settings(tmp_path))[1]

    assert cloudflare.base_url == CLOUDFLARE_BASE_URL.format(account_id="acc")
    assert cloudflare.model == "@cf/openai/gpt-oss-120b"


async def test_first_provider_answers_and_is_named(tmp_path: Path) -> None:
    c, clients = chain(tmp_path, settings(tmp_path))

    result = await c.generate("s", "u")

    assert result.ok and result.text == "from gemini-3.5-flash-lite"
    assert result.provider == "gemini"
    assert sum(client.calls for client in clients.values()) == 1


async def test_spent_daily_quota_moves_to_the_next_provider(tmp_path: Path) -> None:
    c, _ = chain(tmp_path, settings(tmp_path, gemini_daily_limit=1))

    first, second = await c.generate("s", "u"), await c.generate("s", "u")

    assert (first.provider, second.provider) == ("gemini", "cloudflare")


async def test_a_failure_is_not_retried_but_cools_the_provider_down(tmp_path: Path) -> None:
    clock = Clock()
    c, clients = chain(tmp_path, settings(tmp_path), {"gemini": ["rate_limited"]}, clock)

    failed = await c.generate("s", "u")
    after = await c.generate("s", "u")
    clock.now += 121
    recovered = await c.generate("s", "u")

    assert failed.error == "rate_limited" and failed.provider == "gemini"
    assert sum(client.calls for client in clients.values()) == 3  # one call per question
    assert after.provider == "cloudflare"
    assert recovered.provider == "gemini"


async def test_no_provider_left_means_no_call(tmp_path: Path) -> None:
    s = settings(tmp_path, gemini_daily_limit=0, cloudflare_daily_limit=0, openrouter_daily_limit=0)
    c, clients = chain(tmp_path, s)

    result = await c.generate("s", "u")

    assert result.error == "providers_exhausted" and result.usage.llm_calls == 0
    assert all(client.calls == 0 for client in clients.values())


async def test_quota_is_kept_across_restarts(tmp_path: Path) -> None:
    s = settings(tmp_path, gemini_daily_limit=1)
    await chain(tmp_path, s)[0].generate("s", "u")

    result = await chain(tmp_path, s)[0].generate("s", "u")

    assert result.provider == "cloudflare"


async def test_stream_names_the_provider_in_the_result(tmp_path: Path) -> None:
    c, _ = chain(tmp_path, settings(tmp_path))

    items = [item async for item in c.stream("s", "u")]

    assert items[0] == "Hi " and items[-1].provider == "gemini"


async def test_rewrite_chain_is_reused_and_uses_the_rewrite_model(tmp_path: Path) -> None:
    c, _ = chain(tmp_path, settings(tmp_path))

    rewrite = c.for_rewrite()

    assert c.for_rewrite() is rewrite
    assert (await rewrite.generate("s", "u")).text == "from gemini-3.5-flash-lite"
