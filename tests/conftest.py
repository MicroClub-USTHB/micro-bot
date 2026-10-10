import os
from pathlib import Path

import pytest
import structlog

from mcclub_rag.ingest.settings import IngestSettings

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OCR_LANGUAGES = ("ara", "fra", "eng")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # preprocess_* verifies traineddata before parsing anything, so every test that pulls in
    # the tessdata fixture needs the models even if it never OCRs; keep -m "not ocr" honest.
    # Same for the embedder tokenizer: keep -m "not tokenizer" free of the 33 MB download,
    # and for the ONNX model and BM25 stopwords: keep -m "not embedder" free of 313 MB,
    # and -m "not reranker" free of the 341 MB reranker.
    for item in items:
        fixtures = getattr(item, "fixturenames", ())
        if "tessdata_dir" in fixtures:
            item.add_marker(pytest.mark.ocr)
        if "tokenizer_path" in fixtures:
            item.add_marker(pytest.mark.tokenizer)
        if "onnx_path" in fixtures or "real_bm25_dir" in fixtures:
            item.add_marker(pytest.mark.embedder)
        if "reranker_dir" in fixtures:
            item.add_marker(pytest.mark.reranker)


@pytest.fixture(autouse=True)
def _reset_structlog():
    # configure_logging() caches loggers on first use; a logger cached under that config
    # bypasses structlog.testing.capture_logs() in later tests, so restore the defaults.
    yield
    structlog.reset_defaults()


def _tessdata_dir() -> Path:
    env = os.environ.get("TESSDATA_PREFIX")
    return Path(env) if env else REPO / "models" / "tessdata"


@pytest.fixture(scope="session")
def tessdata_dir() -> Path:
    """Traineddata directory. Fails (never skips) when language data is missing."""
    directory = _tessdata_dir()
    missing = [lang for lang in OCR_LANGUAGES if not (directory / f"{lang}.traineddata").is_file()]
    if missing:
        pytest.fail(
            f"OCR traineddata missing in {directory}: {', '.join(missing)}. Run:\n"
            "  uv run python scripts/download_models.py --only tessdata --dest models/tessdata",
            pytrace=False,
        )
    return directory


def _tokenizer_path() -> Path:
    env = os.environ.get("INGEST_CHUNK_TOKENIZER_PATH")
    return Path(env) if env else REPO / "models" / "embedder" / "tokenizer.json"


@pytest.fixture(scope="session")
def tokenizer_path() -> Path:
    """Embedder tokenizer.json. Fails (never skips) when it is missing."""
    path = _tokenizer_path()
    if not path.is_file():
        pytest.fail(
            f"Embedder tokenizer missing at {path}. Run:\n"
            "  uv run python scripts/download_models.py --only tokenizer --dest models/embedder",
            pytrace=False,
        )
    return path


def _models_dir() -> Path:
    env = os.environ.get("MODELS_DIR")
    return Path(env) if env else REPO / "models"


@pytest.fixture(scope="session")
def onnx_path() -> Path:
    """Pinned embedder ONNX model (and its tokenizer). Fails (never skips) when missing."""
    from mcclub_rag.retrieval.registry import GRANITE_R2

    directory = _models_dir() / "embedder"
    path = directory / GRANITE_R2.onnx_file
    if not path.is_file() or not (directory / "tokenizer.json").is_file():
        pytest.fail(
            f"Embedder ONNX model missing at {path}. Run:\n"
            f"  uv run python scripts/download_models.py --only embedder --dest {directory}",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="session")
def reranker_dir() -> Path:
    """Pinned default reranker (gte int8 + tokenizer). Fails (never skips) when missing."""
    from mcclub_rag.retrieval.registry import DEFAULT_RERANKER, RERANKERS

    spec = RERANKERS[DEFAULT_RERANKER]
    root = _models_dir() / "reranker"
    directory = root / spec.slug
    if not (directory / spec.onnx_file).is_file() or not (directory / "tokenizer.json").is_file():
        pytest.fail(
            f"Reranker model missing in {directory}. Run:\n"
            f"  uv run python scripts/download_models.py --only reranker --dest {root}",
            pytrace=False,
        )
    return directory


@pytest.fixture(scope="session")
def real_bm25_dir() -> Path:
    """Pinned BM25 stopword lists. Fails (never skips) when missing."""
    from mcclub_rag.retrieval.registry import BM25_FILES

    directory = _models_dir() / "bm25"
    if not all((directory / name).is_file() for name in BM25_FILES):
        pytest.fail(
            f"BM25 stopword files missing in {directory}. Run:\n"
            f"  uv run python scripts/download_models.py --only bm25 --dest {directory}",
            pytrace=False,
        )
    return directory


@pytest.fixture
def ocr_settings(tessdata_dir: Path) -> IngestSettings:
    return IngestSettings(tessdata_prefix=tessdata_dir)


@pytest.fixture
def fixture_bytes():
    def read(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return read
