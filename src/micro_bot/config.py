from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None
    openrouter_model: str = "google/gemma-4-31b-it:free"
    # Free tier without a card: 50 requests/day, 1000 once 10 credits were ever bought
    # (openrouter.ai/docs/api/reference/limits, 2026-10-18)
    openrouter_daily_limit: int = Field(default=50, ge=0)
    openrouter_rpm: int = Field(default=20, ge=0)  # free models: 20/minute (same docs page)
    cloudflare_rpm: int = Field(default=0, ge=0)  # 0: no per-minute cap applied by the chain
    # Google Gemini API, OpenAI-compatible. Free tier needs no card and lists Algeria
    # (ai.google.dev/gemini-api/docs/available-regions, 2026-10-18); free content may be used
    # to improve Google's products, so only public documents belong here.
    gemini_api_key: str | None = None
    # 2.5 models are closed to new users (HTTP 404, 2026-10-18). Each model accepts different
    # reasoning_effort values, measured: 3.8-flash takes "none" (not "minimal"), 3.5-flash-lite
    # takes "minimal" (not "none"); "" sends none and keeps the model's default thinking.
    # Answers on flash-lite too: 1.0-1.2 s against 3-17 s (one 25 s timeout) for 3.8-flash on
    # the free tier, same answer on the measured questions (2026-10-18)
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_reasoning_effort: str = "minimal"
    # The rewrite on another model: the 15/minute cap is per model, so a question no longer
    # spends two calls of flash-lite's minute. 3.6-flash: 1.3 s median, 5/5 valid JSON on
    # ar/fr/en/arabizi rewrites (2026-10-18), and it takes reasoning_effort "none".
    gemini_rewrite_model: str = "gemini-3.6-flash"
    gemini_rewrite_reasoning_effort: str = "none"
    # Google shows free quotas only in AI Studio: set these to your project's RPD. When the
    # answer and rewrite models are the same they share one counter, so keep the limits equal.
    gemini_daily_limit: int = Field(default=1000, ge=0)
    gemini_rewrite_daily_limit: int = Field(default=1000, ge=0)
    # Free tier requests/minute per project and model (429 quotaId
    # GenerateRequestsPerMinutePerProjectPerModel-FreeTier, measured 2026-10-18): flash-lite 15,
    # the flash models (3.6, 3.8, 3-preview) 5. Past the rewrite's 5, rewrites go to the next
    # provider while answers stay on flash-lite.
    gemini_rpm: int = Field(default=15, ge=0)
    gemini_rewrite_rpm: int = Field(default=5, ge=0)
    # Cloudflare Workers AI: 10,000 free neurons/day; gpt-oss-120b is ~74 neurons per answer
    # (developers.cloudflare.com/workers-ai/platform/pricing, 2026-10-18), so ~130/day
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_model: str = "@cf/openai/gpt-oss-120b"
    cloudflare_daily_limit: int = Field(default=120, ge=0)
    # Order the answer call tries providers in; the first with quota left and no recent
    # failure is used, so a question still makes at most one answer call
    llm_providers: str = "gemini,cloudflare,openrouter"
    llm_provider_cooldown_s: int = Field(default=120, gt=0)  # after a 5xx, timeout or error
    # After a 429: Gemini's says "retry in ~5 s" (per-minute cap); the RPM guard avoids most
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
    # Overall daily cap across providers; each provider also keeps its own *_DAILY_LIMIT
    llm_daily_limit: int = Field(default=1500, gt=0)
    llm_reserve: int = Field(default=5, ge=0)
    llm_timeout_s: int = Field(default=25, gt=0)
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
