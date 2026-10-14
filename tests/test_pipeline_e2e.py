"""Whole pipeline on real models: preprocessing, embedder, BM25, Qdrant, then the reranker."""

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from mcclub_rag.ingest.docstore import DocStore
from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.ingest.settings import IngestSettings
from mcclub_rag.retrieval.backend import HybridRetriever
from mcclub_rag.retrieval.embed import Bm25SparseEncoder, OnnxDenseEncoder
from mcclub_rag.retrieval.registry import DEFAULT_EMBEDDER, EMBEDDERS
from mcclub_rag.retrieval.settings import RetrievalSettings
from tests.conftest import FIXTURES

SPEC = EMBEDDERS[DEFAULT_EMBEDDER]


async def test_markdown_file_end_to_end(
    tmp_path, tessdata_dir, tokenizer_path, onnx_path, real_bm25_dir
):
    settings = IngestSettings(
        tessdata_prefix=tessdata_dir, chunk_tokenizer_path=tokenizer_path, data_dir=tmp_path
    )
    client = AsyncQdrantClient(location=":memory:")
    dense = OnnxDenseEncoder(SPEC, onnx_path.parents[1], dim=SPEC.dim)
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


async def test_retrieve_end_to_end_with_reranker_and_visibility(
    tmp_path, tessdata_dir, tokenizer_path, onnx_path, real_bm25_dir, reranker_dir
):
    settings = IngestSettings(
        tessdata_prefix=tessdata_dir, chunk_tokenizer_path=tokenizer_path, data_dir=tmp_path
    )
    pipeline = IngestPipeline(
        settings,
        dense=OnnxDenseEncoder(SPEC, onnx_path.parents[1], dim=SPEC.dim),
        sparse=Bm25SparseEncoder(real_bm25_dir),
        client=AsyncQdrantClient(location=":memory:"),
        docstore=DocStore(tmp_path / "documents"),
    )
    await pipeline.start()
    public = await pipeline.ingest_file(FIXTURES / "notes.md", "notes.md", "public")
    secret = tmp_path / "board.md"
    secret.write_text(
        "# Board\n\nFor members, the office is open from 8:00 to 20:00 on weekdays.\n",
        encoding="utf-8",
    )
    members = await pipeline.ingest_file(secret, "board.md", "members")
    retriever = HybridRetriever.from_pipeline(
        pipeline,
        RetrievalSettings(models_dir=reranker_dir.parents[1]),
    )

    as_public = await retriever.retrieve("When is the office open?", "public", [])
    assert as_public.decision == "answer"
    assert {c.doc_id for c in as_public.chunks} == {public.doc_id}
    assert "9:00" in as_public.chunks[0].text
    as_member = await retriever.retrieve("When is the office open?", "member", [])
    assert members.doc_id in {c.doc_id for c in as_member.chunks}
    assert as_member.chunks[0].score >= as_member.chunks[-1].score
    await pipeline.close()
