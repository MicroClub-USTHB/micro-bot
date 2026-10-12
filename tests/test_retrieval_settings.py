import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcclub_rag.retrieval.settings import RetrievalSettings, get_retrieval_settings

_SHARED = {
    "RETRIEVE_TOP_K",
    "RERANK_TOP_N",
    "GATE_THRESHOLD",
    "RERANK_MODEL",
    "MODELS_DIR",
    "QDRANT_COLLECTION",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("RETRIEVAL_") or key in _SHARED:
            monkeypatch.delenv(key)
    get_retrieval_settings.cache_clear()
    yield
    get_retrieval_settings.cache_clear()


def test_defaults_match_design():
    s = RetrievalSettings()
    assert s.top_k == 20
    assert s.top_n == 5
    assert s.gate_threshold is None
    assert s.rerank_model == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert s.models_dir == Path("/app/models")
    assert s.qdrant_collection == "mcclub"
    assert s.fusion == "rrf"
    assert s.rrf_k == 2
    assert (s.dense_weight, s.sparse_weight) == (1.0, 1.0)
    assert s.prefetch_limit == 40
    assert s.rerank_max_length == 640
    assert s.rerank_threads == 4
    assert s.followup_max_words == 8
    assert s.followup_cues is True
    assert s.neighbours is False


def test_shared_names_are_read_without_prefix(monkeypatch):
    monkeypatch.setenv("RETRIEVE_TOP_K", "15")
    monkeypatch.setenv("RERANK_TOP_N", "3")
    monkeypatch.setenv("GATE_THRESHOLD", "0.25")
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("MODELS_DIR", "/tmp/models")
    monkeypatch.setenv("QDRANT_COLLECTION", "other")
    monkeypatch.setenv("RETRIEVAL_NEIGHBOURS", "true")
    s = get_retrieval_settings()
    assert (s.top_k, s.top_n, s.gate_threshold) == (15, 3, 0.25)
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.models_dir == Path("/tmp/models")
    assert s.qdrant_collection == "other"
    assert s.neighbours is True


def test_blank_gate_threshold_is_none(monkeypatch):
    monkeypatch.setenv("GATE_THRESHOLD", "")
    assert RetrievalSettings().gate_threshold is None


@pytest.mark.parametrize(
    "env",
    [
        {"RETRIEVAL_PREFETCH_LIMIT": "10", "RETRIEVE_TOP_K": "30"},
        {"RETRIEVAL_RERANK_MAX_LENGTH": "512"},
        {"RETRIEVAL_FUSION": "sum"},
    ],
)
def test_invalid_combinations_raise(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        RetrievalSettings()


def test_rerank_pool_and_cut_are_bounded():
    s = RetrievalSettings()
    assert s.rerank_workers == 1 and s.rerank_top_k is None
    with pytest.raises(ValidationError, match="RERANK_WORKERS"):
        RetrievalSettings(rerank_workers=8, rerank_threads=4)
    with pytest.raises(ValidationError, match="RERANK_TOP_K"):
        RetrievalSettings(rerank_top_k=25)


def test_blank_rerank_top_k_is_none(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_RERANK_TOP_K", "")
    assert RetrievalSettings().rerank_top_k is None
