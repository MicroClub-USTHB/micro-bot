"""Dense (granite ONNX) and sparse (BM25) encoders, shared by ingest and retrieval.

Documents and queries go through the same objects, so the two sides can never drift apart.
See specs/ingest-pipeline/design.md for the measurements behind each choice.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
from fastembed.sparse.bm25 import Bm25
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state
from qdrant_client.models import SparseVector
from tokenizers import Tokenizer

from micro_bot.ingest.errors import EmbedderConfigError, EmbeddingError, IngestError
from micro_bot.ingest.settings import IngestSettings
from micro_bot.retrieval.registry import BM25_FILES, EMBEDDERS, EmbedderSpec
from micro_bot.text.normalize import fold_for_search

SPARSE_VERSION = 1  # bump when folding or BM25 tokenization changes
_DOWNLOAD = "uv run python scripts/download_models.py --only {group} --dest {directory}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_file(
    path: Path,
    expected: str,
    group: str,
    directory: Path,
    error: type[IngestError] = EmbedderConfigError,
) -> None:
    run = _DOWNLOAD.format(group=group, directory=directory)
    if not path.is_file():
        raise error(f"model file missing at {path}. Run: {run}")
    if _sha256_file(path) != expected:
        raise error(f"model file at {path} has the wrong sha256. Run: {run}")


_ORT_ERRORS = (
    ort_state.Fail,
    ort_state.InvalidArgument,
    ort_state.RuntimeException,
    ort_state.EPFail,
    RuntimeError,
    ValueError,
)


class DenseEncoder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OnnxDenseEncoder:
    """granite int8 ONNX: CLS pooling + L2 normalize, no prefix, one sequence per run.

    One sequence per ``session.run``: the int8 model quantizes activations per call, so a
    padded batch changed vectors by up to 0.012 per component, and batch 1 was also faster.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        model_dir: Path,
        *,
        threads: int = 4,
        max_length: int = 512,
        dim: int | None = None,
        session: Any = None,  # tests inject a fake with get_inputs() / run()
    ) -> None:
        if dim is not None and dim != spec.dim:
            raise EmbedderConfigError(f"EMBED_DIM={dim} but {spec.repo} produces {spec.dim}")
        tokenizer_path = model_dir / "tokenizer.json"
        onnx_path = model_dir / spec.onnx_file
        _check_file(tokenizer_path, spec.tokenizer_sha256, "embedder", model_dir)
        _check_file(onnx_path, spec.onnx_sha256, "embedder", model_dir)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length)
        self._tokenizer = tokenizer
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
        self._session = session
        self._inputs = {i.name for i in session.get_inputs()}
        self._dim = spec.dim
        self._spec = spec
        self._fingerprint = f"{spec.repo}@{spec.revision}:{spec.onnx_sha256}"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _embed(self, text: str) -> list[float]:
        ids = np.asarray([self._tokenizer.encode(text).ids], dtype=np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),  # BERT-style exports (e5) ask for it
        }
        try:
            hidden = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})
        except _ORT_ERRORS as exc:
            raise EmbeddingError(f"embedding model failed: {type(exc).__name__}") from exc
        states = np.asarray(hidden[0][0], dtype=np.float32)
        # No padding (one sequence per run): the mean over every token is the masked mean.
        pooled = states.mean(axis=0) if self._spec.pooling == "mean" else states[0]
        norm = float(np.linalg.norm(pooled))
        return (pooled / norm if norm > 0 else pooled).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(self._spec.passage_prefix + text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self._spec.query_prefix + text)


class SparseEncoder(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class Bm25SparseEncoder:
    """Language-agnostic BM25 on ``fold_for_search`` text: no stemming, one ar+fr+en stopword set.

    fastembed compares stopwords with the (already folded) token, so the lists are folded too.
    IDF is applied by Qdrant (``Modifier.IDF``), not here.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 110,
        expected: Mapping[str, str] = BM25_FILES,
    ) -> None:
        words: set[str] = set()
        for name, sha256 in sorted(expected.items()):
            path = model_dir / name
            _check_file(path, sha256, "bm25", model_dir)
            words.update(path.read_text(encoding="utf-8").split())
        stopwords = {fold_for_search(word) for word in words}
        # specific_model_path keeps fastembed from resolving (downloading) Qdrant/bm25.
        self._bm25 = Bm25(
            "Qdrant/bm25",
            specific_model_path=str(model_dir),
            cache_dir=str(model_dir),
            disable_stemmer=True,
            stopwords=stopwords,
            k=k,
            b=b,
            avg_len=avg_len,
        )
        settings = {"version": SPARSE_VERSION, "k": k, "b": b, "avg_len": avg_len}
        settings["stopwords"] = hashlib.sha256("\n".join(sorted(stopwords)).encode()).hexdigest()
        self._fingerprint = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        folded = [fold_for_search(text) for text in texts]
        return [_sparse(e) for e in self._bm25.embed(folded)] if folded else []

    def embed_query(self, text: str) -> SparseVector:
        return _sparse(next(iter(self._bm25.query_embed(fold_for_search(text)))))


def _sparse(embedding) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices], values=[float(v) for v in embedding.values]
    )


def embedder_spec(model: str) -> EmbedderSpec:
    try:
        return EMBEDDERS[model]
    except KeyError:
        known = ", ".join(sorted(EMBEDDERS))
        raise EmbedderConfigError(f"unknown EMBED_MODEL {model!r}; known: {known}") from None


@lru_cache
def _dense(
    model: str, model_dir: Path, threads: int, max_length: int, dim: int
) -> OnnxDenseEncoder:
    spec = embedder_spec(model)
    return OnnxDenseEncoder(spec, model_dir, threads=threads, max_length=max_length, dim=dim)


@lru_cache
def _sparse_encoder(model_dir: Path, k: float, b: float, avg_len: float) -> Bm25SparseEncoder:
    return Bm25SparseEncoder(model_dir, k=k, b=b, avg_len=avg_len)


def get_dense_encoder(settings: IngestSettings) -> OnnxDenseEncoder:
    """One loaded ONNX session per configuration, reused by ingest and retrieval."""
    return _dense(
        settings.embed_model,
        settings.models_dir / "embedder",
        settings.embed_threads,
        settings.embed_max_length,
        settings.embed_dim,
    )


def get_sparse_encoder(settings: IngestSettings) -> Bm25SparseEncoder:
    return _sparse_encoder(
        settings.models_dir / "bm25", settings.bm25_k, settings.bm25_b, settings.bm25_avg_len
    )
