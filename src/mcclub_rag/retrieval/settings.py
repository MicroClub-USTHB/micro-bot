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
    # 20: same MRR as 30 on the eval, rerank 1.3 s instead of 1.9 s (eval/results, 2026-10-10)
    top_k: int = Field(default=20, gt=0, validation_alias="RETRIEVE_TOP_K")
    top_n: int = Field(default=5, gt=0, validation_alias="RERANK_TOP_N")
    # Model-specific (gte: 0.40, bge: ~0.15), so set in the environment, not here.
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
    rrf_k: int = Field(default=2, gt=0)  # 2 beat 60 before rerank: MRR 0.861 vs 0.816
    dense_weight: float = Field(default=1.0, gt=0)
    sparse_weight: float = Field(default=1.0, gt=0)
    prefetch_limit: int = Field(default=40, gt=0)  # per arm
    # Dense arm restricted to chunks NOT in the query's language; 0 = off (eval-fixes Req 10).
    # 20: cross-lingual hit@5 0.755 -> 0.776, overall 0.838 -> 0.851, no group worse (fx-cl20)
    cross_lang_prefetch: int = Field(default=20, ge=0)

    # Reranker
    rerank_max_length: int = MIN_RERANK_MAX_LENGTH
    rerank_threads: int = Field(default=4, gt=0)
    # Reranker sessions used in parallel (eval-fixes Req 17); each adds ~350 MB of RAM (gte)
    rerank_workers: int = Field(default=1, gt=0)
    # Rerank only the first k fused candidates; None = all RETRIEVE_TOP_K
    rerank_top_k: int | None = Field(default=None, gt=0)

    # Query and context
    followup_max_words: int = Field(default=8, ge=0)
    followup_cues: bool = True  # merge only cued messages (eval-fixes Req 11)
    query_fix_case: bool = False  # lower-case an ALL-CAPS query (eval-fixes Req 9)
    query_strip_harakat: bool = False  # drop Arabic diacritics from the query
    neighbours: bool = False
    # Rank multiplier for chunks flagged ``instruction_like`` at ingest; 1.0 = off (Req 4.4)
    instruction_penalty: float = Field(default=1.0, gt=0, le=1)

    @field_validator("gate_threshold", "rerank_top_k", mode="before")
    @classmethod
    def _blank_threshold_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.prefetch_limit < self.top_k:
            raise ValueError("RETRIEVAL_PREFETCH_LIMIT must be at least RETRIEVE_TOP_K")
        if self.rerank_workers > self.rerank_threads:
            raise ValueError("RETRIEVAL_RERANK_WORKERS must be at most RETRIEVAL_RERANK_THREADS")
        if self.rerank_top_k is not None and self.rerank_top_k > self.top_k:
            raise ValueError("RETRIEVAL_RERANK_TOP_K must be at most RETRIEVE_TOP_K")
        if self.rerank_max_length < MIN_RERANK_MAX_LENGTH:
            raise ValueError(f"RETRIEVAL_RERANK_MAX_LENGTH must be >= {MIN_RERANK_MAX_LENGTH}")
        return self


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()
