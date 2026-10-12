"""Limits, paths and thresholds for the ingest side (preprocessing, chunking, pipeline).

Kept separate from the app-wide ``config.Settings`` so the API side can compose it
without either module editing the other's fields. All variables use the ``INGEST_``
prefix, except ``TESSDATA_PREFIX`` (Tesseract's own standard name, set in the Dockerfile)
and the names shared with the app config (``QDRANT_URL``, ``QDRANT_COLLECTION``,
``EMBED_MODEL``, ``EMBED_DIM``, ``MODELS_DIR``, ``DATA_DIR``), which are read as they are.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MB = 1024 * 1024


def _default_tokenizer_path() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/app/models")) / "embedder" / "tokenizer.json"


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        extra="ignore",
        populate_by_name=True,
    )

    # OCR
    tessdata_prefix: Path = Field(
        default=Path("/app/models/tessdata"), validation_alias="TESSDATA_PREFIX"
    )
    ocr_languages: list[str] = ["ara", "fra", "eng"]
    ocr_min_chars_per_page: int = Field(default=32, ge=0)

    # File limits
    max_file_mb: int = Field(default=25, gt=0)
    max_pages: int = Field(default=300, gt=0)
    parse_timeout_s: int = Field(default=60, gt=0)  # hang guard per document
    ocr_page_timeout_s: int = Field(default=10, ge=0)  # extra budget per PDF page / image
    ocr_deskew: bool = True  # straighten skewed photos before OCR (eval-fixes Req 12)
    max_concurrency: int = Field(default=1, gt=0)

    # Web fetching
    web_timeout_s: float = Field(default=15, gt=0)
    web_max_bytes: int = Field(default=5 * _MB, gt=0)
    web_max_redirects: int = Field(default=5, ge=0)
    web_user_agent: str = "mcclub-rag/0.1 (+document ingestion)"

    # Language detection
    lang_codes: list[str] = ["ar", "fr", "en"]
    lang_min_chars: int = Field(default=20, ge=0)
    lang_min_prob: float = Field(default=0.5, ge=0, le=1)
    lang_mixed_share: float = Field(default=0.2, gt=0, le=1)
    lang_default: Literal["ar", "fr", "en"] = "fr"

    # Chunking (sizes in embedder tokens; see specs/chunking/design.md for the measurements)
    chunk_target_tokens: int = Field(default=300, gt=0)
    chunk_max_tokens: int = Field(default=400, gt=0)
    chunk_min_tokens: int = Field(default=80, gt=0)
    chunk_max_header_tokens: int = Field(default=48, gt=0)
    chunk_context_header: bool = True
    chunk_tokenizer_path: Path = Field(default_factory=_default_tokenizer_path)

    # Pipeline: shared names first, then INGEST_* (see specs/ingest-pipeline/design.md)
    qdrant_url: str = Field(default="http://qdrant:6333", validation_alias="QDRANT_URL")
    qdrant_collection: str = Field(default="mcclub", validation_alias="QDRANT_COLLECTION")
    embed_model: str = Field(
        default="ibm-granite/granite-embedding-311m-multilingual-r2",
        validation_alias="EMBED_MODEL",
    )
    embed_dim: int = Field(default=768, gt=0, validation_alias="EMBED_DIM")
    models_dir: Path = Field(default=Path("/app/models"), validation_alias="MODELS_DIR")
    data_dir: Path = Field(default=Path("/data"), validation_alias="DATA_DIR")
    qdrant_timeout_s: float = Field(default=10, gt=0)
    embed_threads: int = Field(default=4, gt=0)
    embed_max_length: int = Field(default=512, gt=0)
    bm25_k: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_avg_len: float = Field(default=110, gt=0)  # measured mean on data/parsed chunks
    upsert_batch: int = Field(default=64, gt=0)
    reindex_on_startup: bool = True

    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> Self:
        if not self.chunk_min_tokens < self.chunk_target_tokens <= self.chunk_max_tokens:
            raise ValueError("chunk sizes must satisfy min < target <= max")
        if self.chunk_max_header_tokens * 4 >= self.chunk_max_tokens:
            raise ValueError("chunk_max_header_tokens must be below chunk_max_tokens / 4")
        if self.embed_max_length < self.chunk_max_tokens + 2:  # + <bos> and room for <eos>
            raise ValueError("embed_max_length must be at least chunk_max_tokens + 2")
        return self

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * _MB


@lru_cache
def get_ingest_settings() -> IngestSettings:
    return IngestSettings()
