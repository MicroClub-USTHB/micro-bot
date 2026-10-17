"""Request/response contracts for the jury API plus the shared nisbenz boundary.

The jury shapes are byte-exact (hidden tests validate them); the boundary
models (IngestResult/Chunk/Retrieval/DocInfo) are the frozen interface both
sides import — changing one needs the other person's PR approval.
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["public", "member"]
Visibility = Literal["public", "members"]
DocStatus = Literal["indexed", "duplicate", "failed"]
HealthStatus = Literal["ok", "degraded"]
FeedbackRating = Literal["up", "down"]
QueryLang = Literal["ar", "fr", "en"]
RetrieveDecision = Literal["answer", "refuse"]


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    conversation_id: str | None = None


class Source(BaseModel):
    doc_id: str
    title: str
    snippet: str


class Usage(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    grounded: bool
    sources: list[Source]
    usage: Usage
    latency_ms: int


class HealthResponse(BaseModel):
    status: HealthStatus
    documents: int


class DocumentCreateResponse(BaseModel):
    doc_id: str


class DocInfo(BaseModel):
    doc_id: str
    title: str
    visibility: Visibility
    status: str
    chunk_count: int
    created_at: str
    topic: str | None = None
    expires_at: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: FeedbackRating
    comment: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool = True


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class IngestResult(BaseModel):
    """Returned by nisbenz's async ingest_file/ingest_url (pipeline derives
    doc_id = uuid5(namespace, topic) from the preprocessing content_hash)."""

    doc_id: str
    title: str
    status: DocStatus
    chunk_count: int
    error: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class Retrieval(BaseModel):
    chunks: list[Chunk]
    top_score: float
    decision: RetrieveDecision
