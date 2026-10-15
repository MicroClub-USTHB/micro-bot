"""Download model assets at image build time (offline at runtime).

Currently: Tesseract traineddata for ara/fra/eng (pinned to tag 4.1.0), the chunking tokenizer
(granite's ``tokenizer.json``), the embedder (``tokenizer.json`` plus the int8 ONNX model,
``EMBED_MODEL``, default bge-m3), the BM25 stopword lists and the reranker (``RERANK_MODEL``,
default bge), all verified by SHA-256. The pins live in ``mcclub_rag.retrieval.registry``.

    uv run python scripts/download_models.py --only tessdata --dest models/tessdata
    uv run python scripts/download_models.py --only tokenizer --dest models/chunk-tokenizer
    uv run python scripts/download_models.py --only embedder --dest models/embedder
    uv run python scripts/download_models.py --only bm25 --dest models/bm25
    uv run python scripts/download_models.py --only reranker --dest models/reranker [--model NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import httpx

from mcclub_rag.retrieval.registry import (
    BM25_FILES,
    BM25_REVISION,
    CHUNK_TOKENIZER,
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    EMBEDDERS,
    RERANKERS,
    EmbedderSpec,
    RerankerSpec,
    bm25_url,
)

Variant = Literal["fast", "best"]

TESSDATA_TAG = "4.1.0"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_{variant}/raw/{tag}/{lang}.traineddata"
TESSDATA_LANGUAGES = ("ara", "fra", "eng")
TESSDATA_SHA256: dict[str, dict[str, str]] = {
    "fast": {
        "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
        "fra": "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca",
        "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    },
    "best": {
        "ara": "ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896",
        "fra": "907743d98915c91a3906dfbf6e48b97598346698fe53aaa797e1a064ffcac913",
        "eng": "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
    },
}

# The embedder of $EMBED_MODEL (default bge-m3).
EMBEDDER = EMBEDDERS[os.environ.get("EMBED_MODEL") or DEFAULT_EMBEDDER]
# The tokenizer that sizes the chunks (granite's, whatever the embedder).
TOKENIZER_REPO = CHUNK_TOKENIZER.repo
TOKENIZER_REVISION = CHUNK_TOKENIZER.revision
TOKENIZER_URL = CHUNK_TOKENIZER.url("tokenizer.json")
TOKENIZER_SHA256 = CHUNK_TOKENIZER.tokenizer_sha256


class DownloadError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_verified(http: httpx.Client, url: str, target: Path, expected: str, label: str) -> None:
    """Stream ``url`` to ``<target>.part``, check its SHA-256, then rename into place.

    A failed or tampered download never leaves a usable-looking file behind.
    """
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise DownloadError(
                f"sha256 mismatch for {label}: expected {expected}, got {digest.hexdigest()}"
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def download_tessdata(
    dest: Path,
    variant: Variant = "fast",
    *,
    languages: Iterable[str] = TESSDATA_LANGUAGES,
    client: httpx.Client | None = None,
    hashes: dict[str, str] | None = None,
) -> list[Path]:
    """Fetch ``<lang>.traineddata`` into ``dest``; skip files already present and valid."""
    expected_hashes = hashes or TESSDATA_SHA256[variant]
    dest.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    written: list[Path] = []
    try:
        for lang in languages:
            target = dest / f"{lang}.traineddata"
            expected = expected_hashes[lang]
            if target.exists() and _sha256_file(target) == expected:
                print(f"tessdata {lang}: already present")
                written.append(target)
                continue
            url = TESSDATA_URL.format(variant=variant, tag=TESSDATA_TAG, lang=lang)
            _fetch_verified(http, url, target, expected, f"{lang}.traineddata ({variant})")
            print(f"tessdata {lang}: downloaded ({variant}, tag {TESSDATA_TAG})")
            written.append(target)
    finally:
        if owns_client:
            http.close()
    return written


def _ensure_file(http: httpx.Client, url: str, target: Path, expected: str, label: str) -> bool:
    """Download ``url`` to ``target`` unless a valid copy is there; True if it downloaded."""
    if target.exists() and _sha256_file(target) == expected:
        print(f"{label}: already present")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    _fetch_verified(http, url, target, expected, label)
    print(f"{label}: downloaded")
    return True


def download_tokenizer(
    dest: Path,
    *,
    client: httpx.Client | None = None,
    sha256: str = TOKENIZER_SHA256,
) -> Path:
    """Fetch the chunking ``tokenizer.json`` into ``dest``; skip it if already valid."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "tokenizer.json"
    if target.exists() and _sha256_file(target) == sha256:
        print("tokenizer.json: already present")
        return target
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        _fetch_verified(http, TOKENIZER_URL, target, sha256, "tokenizer.json")
    finally:
        if owns_client:
            http.close()
    print(f"tokenizer.json: downloaded ({TOKENIZER_REPO}@{TOKENIZER_REVISION[:8]})")
    return target


def download_embedder(
    dest: Path, *, client: httpx.Client | None = None, spec: EmbedderSpec = EMBEDDER
) -> list[Path]:
    """Fetch the embedder's ``tokenizer.json`` and ONNX model into ``dest`` (repo layout)."""
    files = [("tokenizer.json", spec.tokenizer_sha256), (spec.onnx_file, spec.onnx_sha256)]
    http = client or httpx.Client(follow_redirects=True, timeout=600)
    try:
        for name, expected in files:
            label = "tokenizer.json" if name == "tokenizer.json" else "onnx model"
            _ensure_file(http, spec.url(name), dest / name, expected, label)
    finally:
        if client is None:
            http.close()
    print(f"embedder: {spec.repo}@{spec.revision[:8]}")
    return [dest / name for name, _ in files]


def download_bm25(
    dest: Path, *, client: httpx.Client | None = None, files: dict[str, str] | None = None
) -> list[Path]:
    """Fetch the BM25 stopword lists into ``dest`` (fastembed ``specific_model_path``)."""
    pins = BM25_FILES if files is None else files
    http = client or httpx.Client(follow_redirects=True, timeout=120)
    try:
        for name, expected in pins.items():
            _ensure_file(http, bm25_url(name), dest / name, expected, name)
    finally:
        if client is None:
            http.close()
    print(f"bm25 stopwords: Qdrant/bm25@{BM25_REVISION[:8]}")
    return [dest / name for name in pins]


def download_reranker(
    dest: Path,
    *,
    client: httpx.Client | None = None,
    spec: RerankerSpec = RERANKERS[DEFAULT_RERANKER],
) -> list[Path]:
    """Fetch a reranker's ``tokenizer.json`` and ONNX model into ``dest/<slug>/`` (repo layout)."""
    root = dest / spec.slug
    files = [("tokenizer.json", spec.tokenizer_sha256), (spec.onnx_file, spec.onnx_sha256)]
    http = client or httpx.Client(follow_redirects=True, timeout=600)
    try:
        for name, expected in files:
            label = f"reranker {name}"
            _ensure_file(http, spec.url(name), root / name, expected, label)
    finally:
        if client is None:
            http.close()
    print(f"reranker: {spec.repo}@{spec.revision[:8]}")
    return [root / name for name, _ in files]


def _models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", "/app/models"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        action="append",
        choices=["tessdata", "tokenizer", "embedder", "bm25", "reranker"],
        help="asset group to download (repeatable); default: all",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="target directory when downloading a single group "
        "(default: $MODELS_DIR/tessdata, $MODELS_DIR/chunk-tokenizer, $MODELS_DIR/embedder, "
        "$MODELS_DIR/bm25, "
        "$MODELS_DIR/reranker)",
    )
    parser.add_argument(
        "--model",
        choices=sorted(RERANKERS),
        default=os.environ.get("RERANK_MODEL") or DEFAULT_RERANKER,
        help="reranker to download (default: $RERANK_MODEL or bge-reranker-v2-m3)",
    )
    parser.add_argument("--variant", choices=["fast", "best"], default="fast")
    args = parser.parse_args(argv)

    targets = set(args.only or ["tessdata", "tokenizer", "embedder", "bm25", "reranker"])
    if args.dest and len(targets) > 1:
        parser.error("--dest needs exactly one --only group")
    try:
        if "tessdata" in targets:
            download_tessdata(args.dest or _models_dir() / "tessdata", args.variant)
        if "tokenizer" in targets:
            download_tokenizer(args.dest or _models_dir() / "chunk-tokenizer")
        if "embedder" in targets:
            download_embedder(args.dest or _models_dir() / "embedder")
        if "bm25" in targets:
            download_bm25(args.dest or _models_dir() / "bm25")
        if "reranker" in targets:
            download_reranker(args.dest or _models_dir() / "reranker", spec=RERANKERS[args.model])
    except (DownloadError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
