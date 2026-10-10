"""Settings for the read side: hybrid search, reranker, gate (retrieval spec, design).

Kept separate from the app-wide ``config.Settings`` like ``IngestSettings``. Retrieval-only
variables use the ``RETRIEVAL_`` prefix; the names the app config already declares
(``RETRIEVE_TOP_K``, ``RERANK_TOP_N``, ``GATE_THRESHOLD``, ``RERANK_MODEL``, ``MODELS_DIR``,
``QDRANT_COLLECTION``) are read as they are. ``RERANK_MODEL`` defaults to gte here (decided
2026-10-10, see specs/retrieval/requirements.md).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_RERANK_MAX_LENGTH = 640  # chunk cap 400 x 1.28 reranker/granite token ratio + query


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        extra="ignore",
        populate_by_name=True,
    )

    # Shared names (app config declares them too)
    top_k: int = Field(default=30, gt=0, validation_alias="RETRIEVE_TOP_K")
    top_n: int = Field(default=5, gt=0, validation_alias="RERANK_TOP_N")
    gate_threshold: float | None = Field(
        default=None, ge=0, le=1, validation_alias="GATE_THRESHOLD"
    )
    rerank_model: str = Field(
        default="Alibaba-NLP/gte-multilingual-reranker-base", validation_alias="RERANK_MODEL"
    )
    models_dir: Path = Field(default=Path("/app/models"), validation_alias="MODELS_DIR")
    qdrant_collection: str = Field(default="mcclub", validation_alias="QDRANT_COLLECTION")

    # Fusion (Qdrant server-side)
    fusion: Literal["rrf", "dbsf"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)  # Qdrant's own default is 2
    dense_weight: float = Field(default=1.0, gt=0)
    sparse_weight: float = Field(default=1.0, gt=0)
    prefetch_limit: int = Field(default=40, gt=0)  # per arm

    # Reranker
    rerank_max_length: int = MIN_RERANK_MAX_LENGTH
    rerank_threads: int = Field(default=4, gt=0)

    # Query and context
    followup_max_words: int = Field(default=8, ge=0)
    neighbours: bool = False

    @field_validator("gate_threshold", mode="before")
    @classmethod
    def _blank_threshold_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.prefetch_limit < self.top_k:
            raise ValueError("RETRIEVAL_PREFETCH_LIMIT must be at least RETRIEVE_TOP_K")
        if self.rerank_max_length < MIN_RERANK_MAX_LENGTH:
            raise ValueError(f"RETRIEVAL_RERANK_MAX_LENGTH must be >= {MIN_RERANK_MAX_LENGTH}")
        return self


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()
