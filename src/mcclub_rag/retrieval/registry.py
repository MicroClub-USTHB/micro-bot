"""Pinned model files for the encoders (ingest-pipeline Req 4, 5) and rerankers (retrieval Req 4).

No heavy imports: ``scripts/download_models.py`` reads these constants at image build time,
and ``retrieval/embed.py`` checks the same hashes at startup. A later A/B embedder is a new
``EMBEDDERS`` entry plus ``EMBED_MODEL`` in the environment, not a code change elsewhere; the
same goes for ``RERANKERS`` and ``RERANK_MODEL``.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbedderSpec:
    repo: str
    revision: str
    onnx_file: str  # path inside the repo, also the path under $MODELS_DIR/embedder
    onnx_sha256: str
    tokenizer_sha256: str
    dim: int
    pooling: Literal["cls"]

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}"


GRANITE_R2 = EmbedderSpec(
    repo="ibm-granite/granite-embedding-311m-multilingual-r2",
    revision="44399559930365213510b1ee2eb15ded83374f0e",
    onnx_file="onnx/model_quint8_avx2.onnx",
    onnx_sha256="f1fdd44e7e1ac51f12ab7957c7bd092e064d596c288513bf9d326842f669edee",
    tokenizer_sha256="0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f",
    dim=768,
    pooling="cls",
)

EMBEDDERS: dict[str, EmbedderSpec] = {GRANITE_R2.repo: GRANITE_R2}

# Stopword lists only: stemming is off, so nothing else from the repo is needed.
BM25_REPO = "Qdrant/bm25"
BM25_REVISION = "22b8d2af71a76161e18dd432d2cee0eefa66e412"
BM25_FILES: dict[str, str] = {
    "arabic.txt": "0df01c0a184d8c15077c6f0ee70e25e0c0308d827b92b14a19f5f819a0c465d0",
    "french.txt": "58ac7f7c074b70dc0fc86f6fcf40b2c227e9f95627e90e57971d5cce1e30e2e9",
    "english.txt": "019f104ba2ed07436d05f9cdd3383034ad66014edc27fc651f837e1a038b6451",
}


def bm25_url(filename: str) -> str:
    return f"https://huggingface.co/{BM25_REPO}/resolve/{BM25_REVISION}/{filename}"


@dataclass(frozen=True)
class RerankerSpec:
    repo: str
    revision: str
    onnx_file: str
    onnx_sha256: str
    tokenizer_sha256: str

    @property
    def slug(self) -> str:
        """Directory under ``$MODELS_DIR/reranker``, so several rerankers can sit side by side."""
        return self.repo.rsplit("/", 1)[-1]

    def url(self, filename: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}"


# Neither original repo ships ONNX; onnx-community exports, int8 (measured in specs/retrieval).
GTE_RERANKER = RerankerSpec(
    repo="onnx-community/gte-multilingual-reranker-base",
    revision="ee64367e35a2db0da46bb6497e13a18f8bd585cb",
    onnx_file="onnx/model_int8.onnx",
    onnx_sha256="ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35",
    tokenizer_sha256="3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98",
)
BGE_RERANKER = RerankerSpec(
    repo="onnx-community/bge-reranker-v2-m3-ONNX",
    revision="6f5ff65298512715a1e669753bc754d2bc8f367b",
    onnx_file="onnx/model_int8.onnx",
    onnx_sha256="912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
    tokenizer_sha256="8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
)

# Keyed by the original model name (what config.py and .env use) and by the export's repo.
RERANKERS: dict[str, RerankerSpec] = {
    "Alibaba-NLP/gte-multilingual-reranker-base": GTE_RERANKER,
    GTE_RERANKER.repo: GTE_RERANKER,
    "BAAI/bge-reranker-v2-m3": BGE_RERANKER,
    BGE_RERANKER.repo: BGE_RERANKER,
}
DEFAULT_RERANKER = "Alibaba-NLP/gte-multilingual-reranker-base"
