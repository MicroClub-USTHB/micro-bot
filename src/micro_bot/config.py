from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Key pools: a comma-separated list gives each credential its own daily
    # quota, per-minute window and cool-down. The singular GEMINI_API_KEY / OPENROUTER_API_KEY
    # still work and are merged with the list. Gemini counts its free quota per project and Groq
    # per organization, so a key list only multiplies where the keys come from different ones.
    openrouter_api_key: str | None = None
    openrouter_api_keys: str = ""
    # Free models rotate and get pulled (z-ai/glm-4.7 and deepseek-chat-v3 lost their :free slug,
    # answered 404 on 2026-10-29); this one answered every probe, and keeps its thinking in the
    # `reasoning` field instead of the content. The gemma-4 free models are the better Arabic
    # ones but sat at an upstream 429 for a week, so they are the A/B entry, not the default.
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    # Free tier without a card: 50 requests/day, 1000 once 10 credits were ever bought
    # (openrouter.ai/docs/api/reference/limits, 2026-10-18)
    openrouter_daily_limit: int = Field(default=50, ge=0)
    openrouter_rpm: int = Field(default=20, ge=0)  # free models: 20/minute (same docs page)
    # Cloudflare Workers AI: 10,000 free neurons/day per account, charged per model and per
    # token (developers.cloudflare.com/workers-ai/platform/pricing, 2026-10-29). glm-4.7-flash
    # is 5,500 neurons/M in and 36,400/M out, so a 2,500-token prompt with a 400-token answer is
    # ~28 neurons: ~350 answers/day, against ~94 for gpt-oss-120b (31,818/68,182). Thinking is on
    # by default and bills as output, so `_configured` turns it off through chat_template_kwargs.
    # The free plan allows 300 requests/minute per account.
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    # A pool of accounts: "account_id:token,account_id:token" (the free neurons are per account)
    cloudflare_accounts: str = ""
    cloudflare_model: str = "@cf/zai-org/glm-4.7-flash"
    cloudflare_daily_limit: int = Field(default=300, ge=0)
    cloudflare_rpm: int = Field(default=300, ge=0)
    # Groq: no card, OpenAI-compatible (console.groq.com/docs/rate-limits, 2026-10-29). With a
    # grounded answer at ~2,500 tokens the 8000 TPM is what binds, not the 30 RPM: two questions
    # a minute, so the per-minute guard counts tokens and not only requests. A day is 200k TPD
    # (~80 answers) before the 1000 RPD.
    groq_api_key: str | None = None
    groq_api_keys: str = ""
    # Limits are per ORGANIZATION, not per key: several keys of one account share one pool, and
    # only keys of separate accounts multiply it. Models are separate rows in Groq's own table,
    # so a second model id is a second 200k TPD, 8k TPM and 1000 RPD on the same key.
    groq_model: str = "openai/gpt-oss-120b"
    # qwen thinks only when asked, answers darija/arabizi directly, and answers in 0.1-0.5 s
    # against 120b's 0.2-1.5 s; `_GROQ_EFFORT` sends it `none` so the 200k TPD is all answer.
    groq_models: str = "qwen/qwen3.8-27b"
    # 0 would mean "never call this provider" (a spent day), not "unlimited": leave it alone
    groq_daily_limit: int = Field(default=1000, ge=0)
    groq_daily_tokens: int = Field(default=200_000, ge=0)
    groq_rpm: int = Field(default=30, ge=0)
    groq_tpm: int = Field(default=8_000, ge=0)
    # "" = the per-model value in llm/providers.py (`_GROQ_EFFORT`): "low", since gpt-oss bills
    # its thinking as completion tokens and they come out of the same 200k TPD
    groq_reasoning_effort: str = ""
    # Google Gemini API, OpenAI-compatible. Free tier needs no card and lists Algeria
    # (ai.google.dev/gemini-api/docs/available-regions, 2026-10-18); free content may be used
    # to improve Google's products, so only public documents belong here.
    gemini_api_key: str | None = None
    gemini_api_keys: str = ""
    # 2.5 models are closed to new users (HTTP 404, 2026-10-18). Answers on flash-lite:
    # 1.0-1.2 s against 3-17 s (one 25 s timeout) for 3.8-flash on the free tier, same answer
    # on the measured questions (2026-10-18). Free tier 2026-10-21: 15 RPM, 500 RPD per project
    # and model; 3.6/3.8-flash are 5 RPM and only 20 RPD.
    gemini_model: str = "gemini-3.5-flash-lite"
    # "" = the per-model value in llm/providers.py (`_GEMINI_EFFORT`): 3.x cannot stop thinking,
    # `minimal` is its floor, and the wrong value is a 400 (flash-lite rejects "none").
    gemini_reasoning_effort: str = ""
    # The rewrite on 3.1-flash-lite: its own per-model pool, so rewrites never spend the answer
    # model's 500/day, and at 15 RPM instead of the 3.6-flash's 5 (20 RPD made 3.6 a cliff).
    gemini_rewrite_model: str = "gemini-3.1-flash-lite"
    gemini_rewrite_reasoning_effort: str = ""
    # Google shows free quotas only in AI Studio: set these to your project's RPD. When the
    # answer and rewrite models are the same they share one counter, so keep the limits equal.
    gemini_daily_limit: int = Field(default=500, ge=0)
    gemini_rewrite_daily_limit: int = Field(default=500, ge=0)
    # Free tier requests/minute per project and model (429 quotaId
    # GenerateRequestsPerMinutePerProjectPerModel-FreeTier, measured 2026-10-18): flash-lite 15,
    # the flash models (3.6, 3.8, 3-preview) 5.
    gemini_rpm: int = Field(default=15, ge=0)
    gemini_rewrite_rpm: int = Field(default=15, ge=0)
    # Google's daily counters reset at midnight Pacific, not UTC (a UTC reset would hand back
    # "fresh" quota 7-8 h too early and burn 429s all morning)
    gemini_quota_reset_tz: str = "America/Los_Angeles"
    # Order the answer call tries providers in; every credential of a provider is tried before
    # the next provider, so a question still makes at most one answer call
    llm_providers: str = "gemini,groq,cloudflare,openrouter"
    # Which credential of one provider serves the next request: least-used spreads the
    # per-minute caps (two keys = twice the throughput), round-robin sticks for 3 calls,
    # fill-first drains a key's day before touching the next
    llm_key_strategy: str = "least-used"
    llm_provider_cooldown_s: int = Field(default=120, gt=0)  # after a 5xx, timeout or error
    # Escalating cool-down for repeated failures: base * 2**consecutive, capped here
    llm_backoff_max_s: int = Field(default=600, gt=0)
    # After a 429 with no usable retry hint: Gemini's says "retry in ~5 s" (per-minute cap);
    # a daily-quota 429 is not paused, it is taken out until the provider's reset
    llm_rate_limit_cooldown_s: int = Field(default=10, gt=0)
    # Answers to a first message (no history) are reused for the same question and corpus
    answer_cache_size: int = Field(default=512, ge=0)
    api_key: str | None = None  # Bearer token required by /chat and /feedback
    admin_api_key: str | None = None
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "mcclub"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    # Retrieval reads these through RetrievalSettings (retrieval/settings.py); keep them equal (I21)
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    models_dir: Path = Path("/app/models")
    data_dir: Path = Path("/data")
    # Overall daily cap across providers; 0 = the sum of the configured pools, so adding a key
    # raises the cap instead of hitting an old constant
    llm_daily_limit: int = Field(default=0, ge=0)
    llm_reserve: int = Field(default=5, ge=0)
    llm_timeout_s: int = Field(default=25, gt=0)
    # Cap on visible answer tokens; thinking is billed as output, and the provider default lets
    # a reasoning model run long. 0 = the provider's own default.
    llm_max_output_tokens: int = Field(default=800, ge=0)
    # Ask OpenRouter to skip a reasoning model's thinking (nemotron chat 18.1 s -> 3-8 s,
    # 2026-10-16); models that don't reason ignore it
    llm_disable_reasoning: bool = True
    # Query rewrite, the optional first LLM call of a question (specs/query-rewrite)
    rewrite_enabled: bool = True
    rewrite_timeout_s: float = Field(default=6.0, gt=0)
    rewrite_cache_size: int = Field(default=512, ge=0)
    retrieve_top_k: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    gate_threshold: float | None = None
    chat_rate_limit_requests: int = Field(default=20, gt=0)
    chat_rate_limit_window_s: int = Field(default=60, gt=0)
    log_level: str = "INFO"

    @field_validator("gate_threshold", mode="before")
    @classmethod
    def blank_gate_threshold_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("llm_key_strategy")
    @classmethod
    def known_key_strategy(cls, value: str) -> str:
        allowed = {"least-used", "round-robin", "fill-first"}
        strategy = value.strip().lower()
        if strategy not in allowed:
            raise ValueError(f"LLM_KEY_STRATEGY must be one of {sorted(allowed)}")
        return strategy


@lru_cache
def get_settings() -> Settings:
    return Settings()
