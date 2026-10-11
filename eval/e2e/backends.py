"""``DocumentBackend`` and readiness probe over one pipeline instance, for the in-process app.

The app's own ``RealDocumentBackend`` reads the module-level pipeline; the eval builds its own
pipeline (temporary data dir, in-memory Qdrant), so it needs this small adapter. It follows the
same mapping: created / replaced / updated -> ``indexed``, topics read from the docstore.
"""

import tempfile
from pathlib import Path

from fastapi import UploadFile

from mcclub_rag.ingest.pipeline import IngestPipeline
from mcclub_rag.schemas import DocInfo, IngestResult

_STATUS = {"created": "indexed", "replaced": "indexed", "updated": "indexed"}


class PipelineDocuments:
    """``DocumentBackend`` over one pipeline instance (the app's own adapter uses the module's)."""

    def __init__(self, pipeline: IngestPipeline) -> None:
        self.pipeline = pipeline

    async def ingest_file(
        self, file: UploadFile, *, visibility, topic, expires_at=None
    ) -> IngestResult:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / (file.filename or "document")
            path.write_bytes(await file.read())
            result = await self.pipeline.ingest_file(path, path.name, visibility, topic=topic)
        status = _STATUS.get(result.status, result.status)
        return IngestResult(
            doc_id=result.doc_id, title=result.title, status=status, chunk_count=result.chunk_count
        )

    async def list_documents(self) -> list[DocInfo]:
        stored, _ = await self.pipeline.docstore.load_all()
        topics = {item.doc_id: item.topic for item in stored}
        return [
            DocInfo(
                doc_id=d.doc_id,
                title=d.title,
                visibility=d.visibility,
                status="indexed",
                chunk_count=d.chunk_count,
                created_at=d.ingested_at.isoformat(),
                topic=topics.get(d.doc_id),
            )
            for d in await self.pipeline.list_documents()
        ]

    async def delete_document(self, doc_id: str) -> bool:
        return await self.pipeline.delete_document(doc_id)

    async def corpus_version(self) -> str:
        return await self.pipeline.corpus_version()


class Ready:
    def __init__(self, documents: PipelineDocuments | None = None) -> None:
        self.documents = documents

    async def check(self) -> bool:
        if self.documents is None:
            return True
        await self.documents.corpus_version()
        return True
