import hashlib
import importlib.util
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
_spec = importlib.util.spec_from_file_location("download_models", _SCRIPT)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

PAYLOAD = {lang: f"{lang}-model-bytes".encode() * 100 for lang in ("ara", "fra", "eng")}
HASHES = {lang: hashlib.sha256(data).hexdigest() for lang, data in PAYLOAD.items()}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve_payload(request: httpx.Request) -> httpx.Response:
    lang = request.url.path.rsplit("/", 1)[-1].removesuffix(".traineddata")
    return httpx.Response(200, content=PAYLOAD[lang])


def test_downloads_and_verifies_all_languages(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_payload(request)

    paths = dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert sorted(p.name for p in paths) == [
        "ara.traineddata",
        "eng.traineddata",
        "fra.traineddata",
    ]
    for lang, data in PAYLOAD.items():
        assert (tmp_path / f"{lang}.traineddata").read_bytes() == data
    assert all("/tessdata_fast/raw/4.1.0/" in url for url in seen)
    assert not list(tmp_path.glob("*.part"))


def test_hash_mismatch_raises_and_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch for ara"):
        dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert list(tmp_path.iterdir()) == []


def test_http_error_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)
    assert list(tmp_path.iterdir()) == []


def test_existing_valid_file_is_not_downloaded_again(tmp_path):
    for lang, data in PAYLOAD.items():
        (tmp_path / f"{lang}.traineddata").write_bytes(data)

    def handler(request):
        raise AssertionError(f"unexpected download of {request.url}")

    dm.download_tessdata(tmp_path, client=_client(handler), hashes=HASHES)


def test_existing_invalid_file_is_replaced(tmp_path):
    (tmp_path / "ara.traineddata").write_bytes(b"stale")
    dm.download_tessdata(tmp_path, client=_client(_serve_payload), hashes=HASHES)
    assert (tmp_path / "ara.traineddata").read_bytes() == PAYLOAD["ara"]


def test_best_variant_uses_best_repo_and_pins(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_payload(request)

    dm.download_tessdata(tmp_path, "best", client=_client(handler), hashes=HASHES)
    assert all("/tessdata_best/" in url for url in seen)
    assert set(dm.TESSDATA_SHA256["best"]) == {"ara", "fra", "eng"}


def test_cli_passes_dest_and_variant(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dm, "download_tessdata", lambda dest, variant: calls.append((dest, variant))
    )
    rc = dm.main(["--only", "tessdata", "--dest", str(tmp_path), "--variant", "best"])
    assert rc == 0
    assert calls == [(tmp_path, "best")]


def test_cli_default_dest_uses_models_dir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(dm, "download_tessdata", lambda dest, variant: calls.append(dest))
    monkeypatch.setattr(dm, "download_embedder", lambda dest: calls.append(dest))
    monkeypatch.setattr(dm, "download_bm25", lambda dest: calls.append(dest))
    monkeypatch.setattr(dm, "download_reranker", lambda dest, spec: calls.append(dest))
    assert dm.main([]) == 0
    assert calls == [
        tmp_path / "tessdata",
        tmp_path / "embedder",
        tmp_path / "bm25",
        tmp_path / "reranker",
    ]


def test_cli_reports_failure(monkeypatch, capsys):
    def boom(dest, variant):
        raise dm.DownloadError("sha256 mismatch for ara")

    monkeypatch.setattr(dm, "download_tessdata", boom)
    assert dm.main(["--only", "tessdata", "--dest", "/nonexistent"]) == 1
    assert "sha256 mismatch" in capsys.readouterr().err


TOKENIZER = b'{"model": "fake tokenizer"}' * 50
TOKENIZER_HASH = hashlib.sha256(TOKENIZER).hexdigest()


def test_tokenizer_downloaded_from_pinned_revision(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=TOKENIZER)

    path = dm.download_tokenizer(tmp_path, client=_client(handler), sha256=TOKENIZER_HASH)
    assert path == tmp_path / "tokenizer.json"
    assert path.read_bytes() == TOKENIZER
    assert seen == [
        "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2"
        f"/resolve/{dm.TOKENIZER_REVISION}/tokenizer.json"
    ]
    assert not list(tmp_path.glob("*.part"))


def test_tokenizer_hash_mismatch_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch for tokenizer.json"):
        dm.download_tokenizer(tmp_path, client=_client(handler), sha256=TOKENIZER_HASH)
    assert list(tmp_path.iterdir()) == []


def test_existing_valid_tokenizer_is_not_downloaded_again(tmp_path):
    (tmp_path / "tokenizer.json").write_bytes(TOKENIZER)

    def handler(request):
        raise AssertionError(f"unexpected download of {request.url}")

    dm.download_tokenizer(tmp_path, client=_client(handler), sha256=TOKENIZER_HASH)


def test_tokenizer_pin_is_the_audited_hash():
    assert dm.TOKENIZER_SHA256 == (
        "0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f"  # granite
    )


def test_cli_only_tokenizer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dm, "download_tessdata", lambda dest, variant: calls.append("tessdata"))
    monkeypatch.setattr(dm, "download_tokenizer", lambda dest: calls.append(dest))
    assert dm.main(["--only", "tokenizer", "--dest", str(tmp_path)]) == 0
    assert calls == [tmp_path]


def test_cli_dest_with_several_groups_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        dm.main(["--dest", str(tmp_path)])


ONNX = b"onnx-graph-bytes" * 200
SPEC = dm.EmbedderSpec(
    repo="org/model",
    revision="abc123",
    onnx_file="onnx/model_quint8_avx2.onnx",
    onnx_sha256=hashlib.sha256(ONNX).hexdigest(),
    tokenizer_sha256=TOKENIZER_HASH,
    dim=8,
    pooling="cls",
)


def _serve_embedder(request: httpx.Request) -> httpx.Response:
    body = TOKENIZER if request.url.path.endswith("tokenizer.json") else ONNX
    return httpx.Response(200, content=body)


def test_embedder_downloads_tokenizer_and_onnx(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_embedder(request)

    dm.download_embedder(tmp_path, client=_client(handler), spec=SPEC)
    assert (tmp_path / "tokenizer.json").read_bytes() == TOKENIZER
    assert (tmp_path / "onnx" / "model_quint8_avx2.onnx").read_bytes() == ONNX
    assert seen == [
        "https://huggingface.co/org/model/resolve/abc123/tokenizer.json",
        "https://huggingface.co/org/model/resolve/abc123/onnx/model_quint8_avx2.onnx",
    ]


def test_embedder_onnx_mismatch_leaves_no_file(tmp_path):
    def handler(request):
        if request.url.path.endswith("tokenizer.json"):
            return httpx.Response(200, content=TOKENIZER)
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch for onnx"):
        dm.download_embedder(tmp_path, client=_client(handler), spec=SPEC)
    assert list((tmp_path / "onnx").iterdir()) == []


def test_existing_valid_embedder_is_not_downloaded_again(tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "tokenizer.json").write_bytes(TOKENIZER)
    (tmp_path / "onnx" / "model_quint8_avx2.onnx").write_bytes(ONNX)

    def handler(request):
        raise AssertionError(f"unexpected download of {request.url}")

    dm.download_embedder(tmp_path, client=_client(handler), spec=SPEC)


STOPWORDS = {name: f"{name} words".encode() for name in ("arabic.txt", "french.txt")}
STOPWORD_HASHES = {name: hashlib.sha256(data).hexdigest() for name, data in STOPWORDS.items()}


def test_bm25_downloads_pinned_stopword_files(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=STOPWORDS[request.url.path.rsplit("/", 1)[-1]])

    dm.download_bm25(tmp_path, client=_client(handler), files=STOPWORD_HASHES)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == STOPWORDS
    assert all(f"/Qdrant/bm25/resolve/{dm.BM25_REVISION}/" in url for url in seen)


def test_bm25_mismatch_leaves_no_file(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch for arabic.txt"):
        dm.download_bm25(tmp_path, client=_client(handler), files=STOPWORD_HASHES)
    assert list(tmp_path.iterdir()) == []


def test_pins_come_from_the_registry():
    from mcclub_rag.retrieval import registry

    assert dm.EMBEDDER is registry.EMBEDDERS[registry.DEFAULT_EMBEDDER] is registry.BGE_M3
    assert dm.TOKENIZER_SHA256 == registry.CHUNK_TOKENIZER.tokenizer_sha256
    assert registry.CHUNK_TOKENIZER is registry.GRANITE_R2  # chunk sizes measured in granite tokens
    assert set(registry.BM25_FILES) == {"arabic.txt", "french.txt", "english.txt"}


@pytest.mark.parametrize("group", ["embedder", "bm25"])
def test_cli_only_new_groups(monkeypatch, tmp_path, group):
    calls = []
    monkeypatch.setattr(dm, "download_embedder", lambda dest: calls.append(("embedder", dest)))
    monkeypatch.setattr(dm, "download_bm25", lambda dest: calls.append(("bm25", dest)))
    assert dm.main(["--only", group, "--dest", str(tmp_path)]) == 0
    assert calls == [(group, tmp_path)]


RERANKER = dm.RerankerSpec(
    repo="org/reranker-onnx",
    revision="def456",
    onnx_file="onnx/model_int8.onnx",
    onnx_sha256=hashlib.sha256(ONNX).hexdigest(),
    tokenizer_sha256=TOKENIZER_HASH,
)


def test_reranker_downloads_into_its_slug_directory(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return _serve_embedder(request)

    dm.download_reranker(tmp_path, client=_client(handler), spec=RERANKER)
    root = tmp_path / "reranker-onnx"
    assert (root / "tokenizer.json").read_bytes() == TOKENIZER
    assert (root / "onnx" / "model_int8.onnx").read_bytes() == ONNX
    assert seen == [
        "https://huggingface.co/org/reranker-onnx/resolve/def456/tokenizer.json",
        "https://huggingface.co/org/reranker-onnx/resolve/def456/onnx/model_int8.onnx",
    ]


def test_reranker_mismatch_leaves_no_file(tmp_path):
    def handler(request):
        if request.url.path.endswith("tokenizer.json"):
            return httpx.Response(200, content=TOKENIZER)
        return httpx.Response(200, content=b"tampered")

    with pytest.raises(dm.DownloadError, match="sha256 mismatch"):
        dm.download_reranker(tmp_path, client=_client(handler), spec=RERANKER)
    assert list((tmp_path / "reranker-onnx" / "onnx").iterdir()) == []


def test_reranker_keys_resolve_to_one_spec_per_model():
    from mcclub_rag.retrieval import registry

    r = registry.RERANKERS
    assert r["Alibaba-NLP/gte-multilingual-reranker-base"] is registry.GTE_RERANKER
    assert r["onnx-community/gte-multilingual-reranker-base"] is registry.GTE_RERANKER
    assert r["BAAI/bge-reranker-v2-m3"] is r["onnx-community/bge-reranker-v2-m3-ONNX"]
    assert registry.RERANKERS[registry.DEFAULT_RERANKER] is registry.MMARCO_RERANKER
    assert registry.GTE_RERANKER.slug == "gte-multilingual-reranker-base"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        (
            ["--model", "Alibaba-NLP/gte-multilingual-reranker-base"],
            "onnx-community/gte-multilingual-reranker-base",
        ),
    ],
)
def test_cli_only_reranker(monkeypatch, tmp_path, argv, expected):
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    calls = []
    monkeypatch.setattr(dm, "download_reranker", lambda dest, spec: calls.append((dest, spec.repo)))
    assert dm.main(["--only", "reranker", "--dest", str(tmp_path), *argv]) == 0
    assert calls == [(tmp_path, expected)]
