import pytest

from mcclub_rag.ingest import chunk
from mcclub_rag.ingest.chunk import chunking_signature
from mcclub_rag.ingest.settings import IngestSettings
from tests.chunk_helpers import FakeCounter

BASE = IngestSettings()


def test_stable_across_calls():
    assert chunking_signature(BASE, FakeCounter()) == chunking_signature(
        IngestSettings(), FakeCounter()
    )


@pytest.mark.parametrize(
    "update",
    [
        {"chunk_target_tokens": 250},
        {"chunk_max_tokens": 450},
        {"chunk_min_tokens": 60},
        {"chunk_max_header_tokens": 32},
        {"chunk_context_header": False},
    ],
)
def test_changes_with_every_chunk_setting(update):
    changed = BASE.model_copy(update=update)
    assert chunking_signature(changed, FakeCounter()) != chunking_signature(BASE, FakeCounter())


def test_changes_with_the_tokenizer():
    class Other(FakeCounter):
        fingerprint = "other"

    assert chunking_signature(BASE, Other()) != chunking_signature(BASE, FakeCounter())


def test_changes_with_the_algorithm_version(monkeypatch):
    before = chunking_signature(BASE, FakeCounter())
    monkeypatch.setattr(chunk, "CHUNKER_VERSION", chunk.CHUNKER_VERSION + 1)
    assert chunking_signature(BASE, FakeCounter()) != before


def test_ignores_the_tokenizer_path(tmp_path):
    moved = BASE.model_copy(update={"chunk_tokenizer_path": tmp_path / "tokenizer.json"})
    assert chunking_signature(moved, FakeCounter()) == chunking_signature(BASE, FakeCounter())


def test_ignores_non_chunk_settings():
    other = BASE.model_copy(update={"max_pages": 3})
    assert chunking_signature(other, FakeCounter()) == chunking_signature(BASE, FakeCounter())
