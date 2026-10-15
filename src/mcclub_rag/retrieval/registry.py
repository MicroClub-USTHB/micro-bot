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
    pooling: Literal["cls", "mean"]
    query_prefix: str = ""  # e5: "query: " / "passage: "
    passage_prefix: str = ""

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

# Default since 2026-10-13 (eval/results/README.md: darija 0.667 -> 0.771 with the bge reranker).
# BAAI ships no int8 ONNX; onnx-community export, dense head only (CLS).
BGE_M3 = EmbedderSpec(
    repo="onnx-community/bge-m3-ONNX",
    revision="25b9af8e87a38eb120cfe87125383677b9cd309e",
    onnx_file="onnx/model_int8.onnx",
    onnx_sha256="2237f770aad5c71bbc1fc2d361a57f9a37400574cc9eff32626f0cdb49234730",
    tokenizer_sha256="249df0778f236f6ece390de0de746838ef25b9d6954b68c2ee71249e0a9d8fd4",
    dim=1024,
    pooling="cls",
)

# Small A/B entries for the query rewrite (specs/query-rewrite Req 3), 384 dims each.
GRANITE_97M_R2 = EmbedderSpec(
    repo="ibm-granite/granite-embedding-97m-multilingual-r2",
    revision="835ad14087e140460703cf0fae09f97d469d65c2",
    onnx_file="onnx/model_quint8_avx2.onnx",
    onnx_sha256="a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e",
    tokenizer_sha256="4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f",
    dim=384,
    pooling="cls",
)
# intfloat's own ONNX is avx512-vnni only; Xenova's int8 export of the same weights (MIT).
E5_SMALL = EmbedderSpec(
    repo="Xenova/multilingual-e5-small",
    revision="761b726dd34fb83930e26aab4e9ac3899aa1fa78",
    onnx_file="onnx/model_int8.onnx",
    onnx_sha256="4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c",
    tokenizer_sha256="0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    dim=384,
    pooling="mean",
    query_prefix="query: ",
    passage_prefix="passage: ",
)

DEFAULT_EMBEDDER = "BAAI/bge-m3"
# Chunk sizes are counted in granite tokens, whatever the embedder: cutting with bge-m3's tokenizer
# instead lost 5 of 111 eval questions (eval/results/README.md, 2026-10-13).
CHUNK_TOKENIZER = GRANITE_R2

# Keyed by the original model name (what .env uses) and by the export's repo.
EMBEDDERS: dict[str, EmbedderSpec] = {
    GRANITE_R2.repo: GRANITE_R2,
    "BAAI/bge-m3": BGE_M3,
    BGE_M3.repo: BGE_M3,
    GRANITE_97M_R2.repo: GRANITE_97M_R2,
    "intfloat/multilingual-e5-small": E5_SMALL,
    E5_SMALL.repo: E5_SMALL,
}

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
    max_length: int | None = None  # model limit under RETRIEVAL_RERANK_MAX_LENGTH (BERT: 512)

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

# Small A/B entries (query-rewrite Req 3), avx2 int8 exports from the original repos (Apache 2.0).
MMARCO_RERANKER = RerankerSpec(
    repo="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # multilingual, 118M
    revision="1427fd652930e4ba29e8149678df786c240d8825",
    onnx_file="onnx/model_quint8_avx2.onnx",
    onnx_sha256="6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b",
    tokenizer_sha256="62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
    max_length=512,
)
MINILM_RERANKER = RerankerSpec(
    repo="cross-encoder/ms-marco-MiniLM-L6-v2",  # English only, 22M
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    onnx_file="onnx/model_quint8_avx2.onnx",
    onnx_sha256="c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
    tokenizer_sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    max_length=512,
)

# Keyed by the original model name (what config.py and .env use) and by the export's repo.
RERANKERS: dict[str, RerankerSpec] = {
    "Alibaba-NLP/gte-multilingual-reranker-base": GTE_RERANKER,
    GTE_RERANKER.repo: GTE_RERANKER,
    "BAAI/bge-reranker-v2-m3": BGE_RERANKER,
    BGE_RERANKER.repo: BGE_RERANKER,
    MMARCO_RERANKER.repo: MMARCO_RERANKER,
    MINILM_RERANKER.repo: MINILM_RERANKER,
}
# mmarco since 2026-10-15: with the query rewrite, hit@5 0.971 vs bge's 0.974 at 2.7x less rerank
# time (eval/results/README.md). bge (2026-10-13) stays the A/B entry: better when the rewrite
# fails (typed arabizi 0.909 vs 0.818).
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
