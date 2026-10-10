"""Whole pipeline on real models: preprocessing, granite tokenizer + ONNX, BM25, Qdrant."""

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.embed import Bm25SparseEncoder, OnnxDenseEncoder
from mcclub_rag.retrieval.registry import GRANITE_R2
from tests.conftest import FIXTURES


async def test_markdown_file_end_to_end(
    tmp_path, tessdata_dir, tokenizer_path, onnx_path, real_bm25_dir
):
    settings = IngestSettings(
        tessdata_prefix=tessdata_dir, chunk_tokenizer_path=tokenizer_path, data_dir=tmp_path
    )
    client = AsyncQdrantClient(location=":memory:")
    dense = OnnxDenseEncoder(GRANITE_R2, onnx_path.parents[1], dim=768)
    pipeline = IngestPipeline(
        settings,
        dense=dense,
        sparse=Bm25SparseEncoder(real_bm25_dir),
        client=client,
        docstore=DocStore(tmp_path / "documents"),
    )
    assert await pipeline.start() == "created"
    result = await pipeline.ingest_file(FIXTURES / "notes.md", "notes.md", "public")
    assert result.status == "created" and result.chunk_count >= 1

    text = (FIXTURES / "notes.md").read_text(encoding="utf-8")
    query = next(line for line in text.splitlines() if len(line.split()) > 4)
    hits = await client.query_points(
        settings.qdrant_collection,
        prefetch=[
            qm.Prefetch(query=dense.embed_query(query), using="dense", limit=10),
            qm.Prefetch(query=pipeline.sparse.embed_query(query), using="bm25", limit=10),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="visibility", match=qm.MatchValue(value="public"))]
        ),
        limit=3,
    )
    assert hits.points and hits.points[0].payload["doc_id"] == result.doc_id
    assert query.strip().split()[0] in " ".join(p.payload["text"] for p in hits.points)
    await client.close()
