from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from mcclub_rag.deps import require_admin
from mcclub_rag.errors import IngestError
from mcclub_rag.schemas import DocInfo, DocumentCreateResponse, IngestResult, Visibility

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

_INGEST_ERROR_CODES = {
    "unsupported_file_type",
    "file_too_large",
    "url_not_allowed",
    "fetch_error",
    "invalid_input",
    "parse_error",
    "parse_timeout",
    "empty_document",
}


@router.post("", response_model=DocumentCreateResponse)
async def create_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    visibility: Annotated[Visibility, Form()],
    topic: Annotated[str | None, Form()] = None,
    expires_at: Annotated[str | None, Form()] = None,
    _: Annotated[None, Depends(require_admin)] = None,
) -> DocumentCreateResponse:
    result = await request.app.state.mcclub_state.documents.ingest_file(
        file,
        visibility=visibility,
        topic=topic,
        expires_at=expires_at,
    )
    _raise_for_failed_ingest(result)
    return DocumentCreateResponse(doc_id=result.doc_id)


@router.get("", response_model=list[DocInfo])
async def list_documents(
    request: Request,
    _: Annotated[None, Depends(require_admin)] = None,
) -> list[DocInfo]:
    return await request.app.state.mcclub_state.documents.list_documents()


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    request: Request,
    _: Annotated[None, Depends(require_admin)] = None,
) -> dict[str, bool]:
    deleted = await request.app.state.mcclub_state.documents.delete_document(doc_id)
    return {"ok": deleted}


def _raise_for_failed_ingest(result: IngestResult) -> None:
    if result.status != "failed":
        return
    code = result.error if result.error in _INGEST_ERROR_CODES else "invalid_input"
    raise IngestError(code, result.error or "Document ingestion failed.")
