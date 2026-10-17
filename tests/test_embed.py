import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from fastembed.sparse.bm25 import Bm25
from tokenizers import Tokenizer, models, pre_tokenizers

from micro_bot.ingest.errors import EmbedderConfigError, EmbeddingError
from micro_bot.ingest.settings import IngestSettings
from micro_bot.retrieval.embed import Bm25SparseEncoder, OnnxDenseEncoder, get_dense_encoder
from micro_bot.retrieval.registry import EmbedderSpec
from micro_bot.text.normalize import fold_for_search

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


class FakeSession:
    """Stands in for onnxruntime: the CLS state is [3, 4, 0, ...] times the sequence length."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.feeds: list[dict] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, outputs, feed):
        if self.error:
            raise self.error
        self.feeds.append(feed)
        batch, seq = feed["input_ids"].shape
        hidden = np.zeros((batch, seq, 8), dtype=np.float32)
        hidden[:, 0, :2] = [3.0 * seq, 4.0 * seq]
        return [hidden]


@pytest.fixture
def embedder_dir(tmp_path):
    vocab = {"[UNK]": 0, "bonjour": 1, "club": 2, "le": 3}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"fake onnx")
    return tmp_path


def _spec(directory, **overrides) -> EmbedderSpec:
    fields = {
        "repo": "org/fake",
        "revision": "r1",
        "onnx_file": "onnx/model.onnx",
        "onnx_sha256": hashlib.sha256((directory / "onnx/model.onnx").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest(),
        "dim": 8,
        "pooling": "cls",
    }
    return EmbedderSpec(**(fields | overrides))


def test_dense_cls_pooling_and_normalization(embedder_dir):
    session = FakeSession()
    enc = OnnxDenseEncoder(_spec(embedder_dir), embedder_dir, session=session)
    vectors = enc.embed_documents(["le club", "bonjour"])
    assert [v[:2] for v in vectors] == [pytest.approx([0.6, 0.8])] * 2
    assert enc.embed_query("le club") == vectors[0]
    assert [f["input_ids"].shape for f in session.feeds] == [(1, 2), (1, 1), (1, 2)]
    assert enc.dim == 8
    assert enc.fingerprint.startswith("org/fake@r1:")


def test_dense_truncates_to_max_length(embedder_dir):
    session = FakeSession()
    enc = OnnxDenseEncoder(_spec(embedder_dir), embedder_dir, max_length=3, session=session)
    enc.embed_query("le club " * 10)
    assert session.feeds[0]["input_ids"].shape == (1, 3)


def test_missing_onnx_fails_with_instruction(embedder_dir):
    spec = _spec(embedder_dir)
    (embedder_dir / "onnx" / "model.onnx").unlink()
    with pytest.raises(EmbedderConfigError, match="--only embedder"):
        OnnxDenseEncoder(spec, embedder_dir, session=FakeSession())


def test_wrong_onnx_hash_fails(embedder_dir):
    with pytest.raises(EmbedderConfigError, match="wrong sha256"):
        OnnxDenseEncoder(_spec(embedder_dir, onnx_sha256="0" * 64), embedder_dir)


def test_dim_mismatch_fails(embedder_dir):
    with pytest.raises(EmbedderConfigError, match="EMBED_DIM=768"):
        OnnxDenseEncoder(_spec(embedder_dir), embedder_dir, dim=768, session=FakeSession())


def test_unknown_model_fails():
    with pytest.raises(EmbedderConfigError, match="unknown EMBED_MODEL"):
        get_dense_encoder(IngestSettings(embed_model="org/unknown"))


def test_inference_failure_becomes_embedding_error(embedder_dir):
    session = FakeSession(error=RuntimeError("bad input"))
    enc = OnnxDenseEncoder(_spec(embedder_dir), embedder_dir, session=session)
    with pytest.raises(EmbeddingError, match="RuntimeError"):
        enc.embed_documents(["le club"])


def test_dense_mean_pooling_and_prefixes(embedder_dir):
    session = FakeSession()
    spec = _spec(embedder_dir, pooling="mean", query_prefix="le ", passage_prefix="bonjour ")
    enc = OnnxDenseEncoder(spec, embedder_dir, session=session)
    doc = enc.embed_documents(["club"])[0]
    query = enc.embed_query("club")
    # prefixes add one token each: "bonjour club" / "le club"
    assert [f["input_ids"].tolist() for f in session.feeds] == [[[1, 2]], [[3, 2]]]
    # only the first token is non-zero: the mean has the same direction as CLS here
    assert doc[:2] == pytest.approx([0.6, 0.8]) and query[:2] == pytest.approx([0.6, 0.8])
