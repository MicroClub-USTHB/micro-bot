"""Dense (granite ONNX) and sparse (BM25) encoders, shared by ingest and retrieval.

Documents and queries go through the same objects, so the two sides can never drift apart.
See specs/ingest-pipeline/design.md for the measurements behind each choice.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from fastembed.sparse.bm25 import Bm25
from qdrant_client.models import SparseVector

from mcclub_rag.ingest.errors import EmbedderConfigError
from mcclub_rag.retrieval.registry import BM25_FILES
from mcclub_rag.text.normalize import fold_for_search

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


def _check_file(path: Path, expected: str, group: str, directory: Path) -> None:
    run = _DOWNLOAD.format(group=group, directory=directory)
    if not path.is_file():
        raise EmbedderConfigError(f"model file missing at {path}. Run: {run}")
    if _sha256_file(path) != expected:
        raise EmbedderConfigError(f"model file at {path} has the wrong sha256. Run: {run}")


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
