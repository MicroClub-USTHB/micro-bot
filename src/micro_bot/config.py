from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None
    openrouter_model: str = "google/gemma-4-31b-it:free"
    member_token: str | None = None
    admin_api_key: str | None = None
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "mcclub"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    # Retrieval reads these through RetrievalSettings (retrieval/settings.py); keep them equal (I21)
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    models_dir: Path = Path("/app/models")
    data_dir: Path = Path("/data")
    llm_daily_limit: int = Field(default=50, gt=0)
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
