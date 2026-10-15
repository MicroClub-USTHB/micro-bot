"""Cross-encoder reranking (retrieval Req 4): ONNX int8, one pair per run, sigmoid scores."""

import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from mcclub_rag.ingest.errors import RerankerConfigError, RerankError
from mcclub_rag.retrieval.embed import _ORT_ERRORS, _check_file
from mcclub_rag.retrieval.registry import RERANKERS, RerankerSpec
from mcclub_rag.retrieval.settings import RetrievalSettings
from mcclub_rag.text.normalize import drop_link_targets


def passage(title: str, heading_path: Sequence[str], text: str, *, header: bool = True) -> str:
    """Reranker passage: the 'title > heading path' header the chunk was embedded with, then text.

    Same rule as ``ingest/chunk.py::_header`` minus its token trim: the reranker truncates the
    passage side only, so a long header just leaves less room for the text. Link targets are
    dropped as in ``embed_text`` (I30).
    """
    text = drop_link_targets(text)
    entries: list[str] = []
    for entry in (title, *heading_path) if header else ():
        entry = entry.strip()
        if entry and (not entries or entries[-1].casefold() != entry.casefold()):
            entries.append(entry)
    return f"{' > '.join(entries)}\n\n{text}" if entries else text


class Reranker(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class OnnxReranker:
    """Cross-encoder int8 ONNX, **one pair per ``session.run``**, scores = sigmoid(logit).

    Batching is out: int8 quantizes activations per input tensor, so a pair's logit moved by up
    to 1.8 when scored in a padded batch (and batches were slower on CPU). See
    specs/retrieval/design.md.
    """

    def __init__(
        self,
        spec: RerankerSpec,
        model_dir: Path,
        *,
        threads: int = 4,
        max_length: int = 640,
        session: Any = None,  # tests inject a fake with get_inputs() / run()
    ) -> None:
        tokenizer_path = model_dir / "tokenizer.json"
        onnx_path = model_dir / spec.onnx_file
        for path, expected in (
            (tokenizer_path, spec.tokenizer_sha256),
            (onnx_path, spec.onnx_sha256),
        ):
            _check_file(path, expected, "reranker", model_dir.parent, RerankerConfigError)
        # gte's tokenizer.json ships fixed-length padding: turn it off. Truncate the passage
        # only; a query too long for that falls back to trimming both sides.
        self._tokenizers = []
        for strategy in ("only_second", "longest_first"):
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.no_padding()
            tokenizer.enable_truncation(max_length, strategy=strategy)
            self._tokenizers.append(tokenizer)
        warm = session is None
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
        self._session = session
        self._inputs = {i.name for i in session.get_inputs()}
        self._fingerprint = f"{spec.repo}@{spec.revision}:{spec.onnx_sha256}"
        if warm:  # first run pays allocator setup; keep it off the first chat request
            self.score("warm up", ["warm up"])

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _encode(self, query: str, text: str):
        try:
            return self._tokenizers[0].encode(query, text)
        except Exception:  # tokenizers raises a bare Exception when the query alone is too long
            return self._tokenizers[1].encode(query, text)

    def _logit(self, query: str, text: str) -> float:
        ids = np.asarray([self._encode(query, text).ids], dtype=np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }
        try:
            out = self._session.run(None, {k: v for k, v in feed.items() if k in self._inputs})
        except _ORT_ERRORS as exc:
            raise RerankError(f"reranker failed: {type(exc).__name__}") from exc
        return float(np.asarray(out[0]).reshape(-1)[0])

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [_sigmoid(self._logit(query, text)) for text in passages]


def reranker_spec(model: str) -> RerankerSpec:
    try:
        return RERANKERS[model]
    except KeyError:
        known = ", ".join(sorted(RERANKERS))
        raise RerankerConfigError(f"unknown RERANK_MODEL {model!r}; known: {known}") from None


@lru_cache
def _reranker(
    model: str, root: Path, threads: int, max_length: int, worker: int = 0
) -> OnnxReranker:
    spec = reranker_spec(model)
    return OnnxReranker(spec, root / spec.slug, threads=threads, max_length=max_length)


def get_reranker(settings: RetrievalSettings, worker: int = 0) -> OnnxReranker:
    """One loaded ONNX session per configuration and worker.

    With ``RETRIEVAL_RERANK_WORKERS`` > 1 each session gets its share of
    ``RETRIEVAL_RERANK_THREADS``, so the pool uses the same cores as one session.
    """
    return _reranker(
        settings.rerank_model,
        settings.models_dir / "reranker",
        max(1, settings.rerank_threads // settings.rerank_workers),
        settings.rerank_max_length,
        worker,
    )
