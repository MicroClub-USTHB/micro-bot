import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcclub_rag.ingest.settings import IngestSettings, get_ingest_settings

_SHARED = {
    "TESSDATA_PREFIX",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBED_MODEL",
    "EMBED_DIM",
    "MODELS_DIR",
    "DATA_DIR",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("INGEST_") or key in _SHARED:
            monkeypatch.delenv(key)
    get_ingest_settings.cache_clear()
    yield
    get_ingest_settings.cache_clear()


def test_defaults_match_design():
    s = IngestSettings()
    assert s.tessdata_prefix == Path("/app/models/tessdata")
    assert s.ocr_languages == ["ara", "fra", "eng"]
    assert s.ocr_min_chars_per_page == 32
    assert s.max_file_mb == 25
    assert s.max_file_bytes == 25 * 1024 * 1024
    assert s.max_pages == 300
    assert s.parse_timeout_s == 60
    assert s.ocr_page_timeout_s == 10
    assert s.max_concurrency == 1
    assert s.web_timeout_s == 15
    assert s.web_max_bytes == 5 * 1024 * 1024
    assert s.web_max_redirects == 5
    assert s.web_user_agent.startswith("mcclub-rag/")
    assert s.lang_codes == ["ar", "fr", "en"]
    assert s.lang_min_chars == 20
    assert s.lang_min_prob == 0.5
    assert s.lang_mixed_share == 0.2
    assert s.lang_default == "fr"


def test_prefixed_env_overrides_default(monkeypatch):
    monkeypatch.setenv("INGEST_MAX_PAGES", "5")
    monkeypatch.setenv("INGEST_WEB_TIMEOUT_S", "2.5")
    s = IngestSettings()
    assert s.max_pages == 5
    assert s.web_timeout_s == 2.5


def test_list_env_override_uses_json(monkeypatch):
    monkeypatch.setenv("INGEST_OCR_LANGUAGES", '["eng"]')
    assert IngestSettings().ocr_languages == ["eng"]


def test_unprefixed_name_is_ignored(monkeypatch):
    monkeypatch.setenv("MAX_PAGES", "5")
    assert IngestSettings().max_pages == 300


def test_tessdata_prefix_read_from_standard_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert IngestSettings().tessdata_prefix == tmp_path


def test_tessdata_prefix_settable_by_field_name(tmp_path):
    assert IngestSettings(tessdata_prefix=tmp_path).tessdata_prefix == tmp_path


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("INGEST_MAX_PAGES", "0"),
        ("INGEST_MAX_FILE_MB", "-1"),
        ("INGEST_LANG_DEFAULT", "de"),
        ("INGEST_LANG_MIXED_SHARE", "1.5"),
        ("INGEST_LANG_MIN_PROB", "-0.1"),
    ],
)
def test_invalid_values_rejected(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        IngestSettings()


def test_get_ingest_settings_is_cached():
    assert get_ingest_settings() is get_ingest_settings()


def test_chunk_defaults_match_design(monkeypatch):
    monkeypatch.delenv("MODELS_DIR", raising=False)
    s = IngestSettings()
    assert s.chunk_target_tokens == 300
    assert s.chunk_max_tokens == 400
    assert s.chunk_min_tokens == 80
    assert s.chunk_max_header_tokens == 48
    assert s.chunk_context_header is True
    assert s.chunk_tokenizer_path == Path("/app/models/chunk-tokenizer/tokenizer.json")


def test_chunk_tokenizer_path_follows_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert IngestSettings().chunk_tokenizer_path == tmp_path / "chunk-tokenizer" / "tokenizer.json"


def test_chunk_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_CHUNK_MAX_TOKENS", "500")
    monkeypatch.setenv("INGEST_CHUNK_CONTEXT_HEADER", "false")
    monkeypatch.setenv("INGEST_CHUNK_TOKENIZER_PATH", str(tmp_path / "t.json"))
    s = IngestSettings()
    assert s.chunk_max_tokens == 500
    assert s.chunk_context_header is False
    assert s.chunk_tokenizer_path == tmp_path / "t.json"


@pytest.mark.parametrize(
    "fields",
    [
        {"chunk_min_tokens": 300},  # min == target
        {"chunk_min_tokens": 350},  # min > target
        {"chunk_target_tokens": 450},  # target > max
        {"chunk_max_header_tokens": 100},  # header >= max / 4
        {"chunk_min_tokens": 0},
    ],
)
def test_chunk_size_ordering_enforced(fields):
    with pytest.raises(ValidationError):
        IngestSettings(**fields)


def test_pipeline_defaults_match_design():
    s = IngestSettings()
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.qdrant_collection == "mcclub"
    assert s.qdrant_timeout_s == 10
    assert s.embed_model == "BAAI/bge-m3"
    assert s.embed_dim == 1024
    assert s.models_dir == Path("/app/models")
    assert s.data_dir == Path("/data")
    assert s.embed_threads == 4
    assert s.embed_max_length == 512
    assert (s.bm25_k, s.bm25_b, s.bm25_avg_len) == (1.2, 0.75, 110)
    assert s.upsert_batch == 64
    assert s.reindex_on_startup is True


def test_shared_names_read_without_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "m"))
    s = IngestSettings()
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "test"
    assert s.data_dir == tmp_path
    assert s.models_dir == tmp_path / "m"


def test_pipeline_env_override(monkeypatch):
    monkeypatch.setenv("INGEST_EMBED_THREADS", "2")
    monkeypatch.setenv("INGEST_REINDEX_ON_STARTUP", "false")
    s = IngestSettings()
    assert s.embed_threads == 2
    assert s.reindex_on_startup is False


def test_embed_max_length_must_fit_chunks(monkeypatch):
    monkeypatch.setenv("INGEST_EMBED_MAX_LENGTH", "300")
    with pytest.raises(ValidationError):
        IngestSettings()
    with pytest.raises(ValidationError):
        IngestSettings(embed_max_length=401)  # chunk_max_tokens + 2 = 402
