from fastapi import APIRouter, Request

from mcclub_rag.schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state.mcclub_state
    documents = 0
    document_ready = False
    provider_ready = False

    try:
        documents = len(await state.documents.list_documents())
        document_ready = await state.document_probe.check()
    except Exception:
        document_ready = False

    try:
        provider_ready = await state.provider_probe.check()
    except Exception:
        provider_ready = False

    return HealthResponse(
        status="ok" if document_ready and provider_ready else "degraded",
        documents=documents,
    )
