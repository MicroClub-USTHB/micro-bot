"""Chat endpoint assembled exclusively from injectable application boundaries."""

from time import perf_counter
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from mcclub_rag.answer import build, extractive_fallback
from mcclub_rag.conversation import Conversation, Turn
from mcclub_rag.deps import get_role
from mcclub_rag.llm.client import GenerationUsage
from mcclub_rag.llm.prompts import build_prompt
from mcclub_rag.logging import set_request_log_context
from mcclub_rag.schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    QueryLang,
    Retrieval,
    Role,
    Source,
    Usage,
)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    role: Annotated[Role, Depends(get_role)],
) -> ChatResponse:
    state = request.app.state.mcclub_state
    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many chat requests. Please try again shortly.",
        )

    started_at = perf_counter()
    has_conversation_param = payload.conversation_id is not None
    previous = (
        state.conversations.load(payload.conversation_id) if has_conversation_param else None
    )
    language = (
        state.conversations.reuse_language(payload.conversation_id, payload.message)
        if has_conversation_param
        else None
    )
    if language is None:
        language = _query_language(state.language_detector.detect(payload.message))
    conversation = previous or state.conversations.create(language)
    history = previous.messages if previous else []

    state.corpus_version = await state.documents.corpus_version()
    # Skip cache when conversation history is present or when the request carried a conversation_id
    skip_cache = has_conversation_param or bool(history)
    cached = None if skip_cache else state.chat_cache.get(payload.message, role)
    if cached is not None:
        response = cached.model_copy(
            update={
                "conversation_id": conversation.conversation_id,
                "usage": Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
                "latency_ms": _latency_ms(started_at),
            }
        )
        _save_and_log(
            request,
            payload.message,
            response,
            role,
            state.corpus_version,
            conversation,
            [],
        )
        return response

    retrieval = await state.retrieval.retrieve(payload.message, role, history)
    response = await _answer(
        payload.message,
        retrieval,
        language,
        conversation.conversation_id,
        state,
        started_at,
        history=history,
    )
    if not skip_cache:
        state.chat_cache.set(payload.message, role, response)
    _save_and_log(
        request,
        payload.message,
        response,
        role,
        state.corpus_version,
        conversation,
        retrieval.chunks,
    )
    return response


async def _answer(
    message: str,
    retrieval: Retrieval,
    language: QueryLang,
    conversation_id: str,
    state: object,
    started_at: float,
    history: list[Turn] | None = None,
) -> ChatResponse:
    if retrieval.decision == "refuse":
        return _response(
            "I don't know based on the available documents.",
            conversation_id,
            False,
            [],
            Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
            started_at,
        )

    if not state.breaker.allow() or not state.quota.acquire():
        answer, sources, grounded = extractive_fallback(retrieval.chunks)
        return _response(
            answer,
            conversation_id,
            grounded,
            sources,
            Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
            started_at,
        )

    system, user = build_prompt(message, language, retrieval.chunks, history=history or [])
    generation = await state.generator.generate(system, user)
    if generation.ok:
        state.breaker.record_success()
        answer, sources, grounded = build(generation.text or "", retrieval.chunks)
    else:
        state.breaker.record_failure()
        answer, sources, grounded = extractive_fallback(retrieval.chunks)
    return _response(answer, conversation_id, grounded, sources, generation.usage, started_at)


def _response(
    answer: str,
    conversation_id: str,
    grounded: bool,
    sources: list[Source],
    usage: Usage | GenerationUsage,
    started_at: float,
) -> ChatResponse:
    response_usage = Usage(
        llm_calls=usage.llm_calls,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        grounded=grounded,
        sources=sources,
        usage=response_usage,
        latency_ms=_latency_ms(started_at),
    )


def _save_and_log(
    request: Request,
    message: str,
    response: ChatResponse,
    role: Role,
    corpus_version: str,
    conversation: Conversation,
    chunks: list[Chunk],
) -> None:
    source_ids = {source.doc_id for source in response.sources}
    chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.doc_id in source_ids]
    request.app.state.mcclub_state.conversations.save(
        conversation.conversation_id,
        Turn(user_message=message, assistant_message=response.answer),
        chunk_ids,
    )
    set_request_log_context(
        request,
        role=role,
        llm_calls=response.usage.llm_calls,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        corpus_version=corpus_version,
        conversation_id=response.conversation_id,
    )


def _query_language(value: str) -> QueryLang:
    if value not in {"ar", "fr", "en"}:
        raise ValueError("language detector must return ar, fr, or en")
    return cast(QueryLang, value)


def _latency_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
