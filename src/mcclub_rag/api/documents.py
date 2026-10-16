from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

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


@router.put("", response_model=DocumentCreateResponse)
async def replace_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    replaces: Annotated[str, Form(min_length=1)],
    visibility: Annotated[Visibility | None, Form()] = None,
    expires_at: Annotated[str | None, Form()] = None,
    _: Annotated[None, Depends(require_admin)] = None,
) -> DocumentCreateResponse:
    """Replace an existing document with a new file, whatever the new file is called.

    ``replaces`` is the exact topic of the document (its file name without extension unless an
    explicit topic was given), as shown by ``GET /api/v1/documents``. An unknown name is a 404 and
    nothing is ingested, so a typo can never leave two versions in the index.
    """
    documents = request.app.state.mcclub_state.documents
    existing = _find_by_name(await documents.list_documents(), replaces)
    if existing is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No document named '{replaces}' to replace."
        )
    result = await documents.ingest_file(
        file,
        visibility=visibility or existing.visibility,
        topic=existing.topic,
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


def _find_by_name(documents: list[DocInfo], name: str) -> DocInfo | None:
    """The document whose topic is ``name``, or ``name`` minus its extension (``bylaws.pdf``)."""
    for candidate in (name, PurePath(name).stem):
        for doc in documents:
            if doc.topic == candidate:
                return doc
    return None
