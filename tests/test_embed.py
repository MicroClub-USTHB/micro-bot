import hashlib

import pytest
from fastembed.sparse.bm25 import Bm25

from mcclub_rag.ingest.errors import EmbedderConfigError
from mcclub_rag.retrieval.embed import Bm25SparseEncoder
from mcclub_rag.text.normalize import fold_for_search

STOPWORDS = {"arabic.txt": "إلى\nفي\n", "french.txt": "le\nla\ndu\n", "english.txt": "the\nof\n"}


@pytest.fixture
def bm25_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for name, text in STOPWORDS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _hashes(directory) -> dict[str, str]:
    return {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in STOPWORDS}


def _encoder(directory, **kwargs) -> Bm25SparseEncoder:
    return Bm25SparseEncoder(directory, expected=_hashes(directory), **kwargs)


def _token(word: str) -> int:
    return Bm25.compute_token_id(fold_for_search(word))


def test_arabic_hamza_variants_match(bm25_dir):
    enc = _encoder(bm25_dir)
    doc = enc.embed_documents(["موعد الإنتخابات في النادي"])[0]
    query = enc.embed_query("الانتخابات")
    assert set(query.indices) <= set(doc.indices)


def test_accents_and_case_fold(bm25_dir):
    enc = _encoder(bm25_dir)
    doc = enc.embed_documents(["Le Règlement intérieur du club"])[0]
    assert _token("reglement") in doc.indices
    assert set(enc.embed_query("REGLEMENT interieur").indices) <= set(doc.indices)


def test_folded_stopwords_are_dropped(bm25_dir):
    enc = _encoder(bm25_dir)
    doc = enc.embed_documents(["ذهب إلى النادي the club"])[0]
    assert _token("إلى") not in doc.indices  # arrives folded as الي
    assert _token("the") not in doc.indices
    assert _token("النادي") in doc.indices


def test_empty_inputs(bm25_dir):
    enc = _encoder(bm25_dir)
    assert enc.embed_documents([]) == []
    assert enc.embed_documents(["   "])[0].indices == []


def test_fingerprint_tracks_settings(bm25_dir):
    base = _encoder(bm25_dir).fingerprint
    assert base == _encoder(bm25_dir).fingerprint
    assert base != _encoder(bm25_dir, avg_len=200).fingerprint


def test_missing_stopword_file_fails(bm25_dir):
    expected = _hashes(bm25_dir)
    (bm25_dir / "french.txt").unlink()
    with pytest.raises(EmbedderConfigError, match="--only bm25"):
        Bm25SparseEncoder(bm25_dir, expected=expected)


def test_wrong_stopword_hash_fails(bm25_dir):
    expected = _hashes(bm25_dir) | {"arabic.txt": "0" * 64}
    with pytest.raises(EmbedderConfigError, match="wrong sha256"):
        Bm25SparseEncoder(bm25_dir, expected=expected)
