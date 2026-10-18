"""Chat endpoint assembled exclusively from injectable application boundaries.

``stream=false`` (default) answers with one JSON ``ChatResponse``. ``stream=true`` answers with
Server-Sent Events: ``meta`` (``conversation_id``), ``delta`` events (``text``, the answer as
the model writes it), then ``done`` (the full ``ChatResponse``, authoritative) or ``error``
(``{"error": {"code", "message"}}``). Auth and the rate limit are checked first, so they still
answer with an HTTP status and the JSON error.
"""

import json
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Annotated, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event

from micro_bot.answer import (
    StreamFilter,
    build,
    extractive_fallback,
    leaks_prompt,
    strip_references,
)
from micro_bot.answer_cache import cache_key
from micro_bot.conversation import Conversation, Turn
from micro_bot.deps import require_api_key
from micro_bot.llm.client import GenerationResult, GenerationUsage
from micro_bot.llm.prompts import build_prompt
from micro_bot.llm.rewrite import RewriteOutcome
from micro_bot.logging import set_request_log_context
from micro_bot.messages import EMPTY_MESSAGE, NO_ANSWER
from micro_bot.schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    QueryLang,
    Retrieval,
    Source,
    Usage,
)
from micro_bot.text.lang import detect_language

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post(
    "",
    response_model=ChatResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "stream=true"}},
)
async def chat(
    payload: ChatRequest,
    request: Request,
    _: Annotated[None, Depends(require_api_key)] = None,
) -> ChatResponse | StreamingResponse:
    state = request.app.state.micro_bot_state
    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many chat requests. Please try again shortly.",
        )

    started_at = perf_counter()
    has_conversation_param = payload.conversation_id is not None
    previous = state.conversations.load(payload.conversation_id) if has_conversation_param else None
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
    if payload.stream:
        events = _events(request, payload.message, state, conversation, history, started_at)
        return StreamingResponse(events, media_type="text/event-stream", headers=_SSE_HEADERS)
    if not payload.message.strip():  # I18: nothing to search; the API shape stays the same
        retrieval = Retrieval(chunks=[], top_score=0.0, decision="refuse")
        response = _response(
            EMPTY_MESSAGE[language],
            conversation.conversation_id,
            False,
            [],
            Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
            started_at,
        )
        _save_and_log(request, payload.message, response, state.corpus_version, conversation, [])
        return response

    cached = _cached(state, payload.message, history, language, conversation, started_at)
    if cached is not None:
        _save_and_log(request, payload.message, cached, state.corpus_version, conversation, [])
        return cached
    rewrite = await _rewrite(state, payload.message, history)
    retrieval = await state.retrieval.retrieve(payload.message, history, rewrite=rewrite.rewrite)
    response = await _answer(
        payload.message,
        retrieval,
        language,
        conversation.conversation_id,
        state,
        started_at,
        history=history,
    )
    response = _add_usage(response, rewrite.usage)
    _remember(state, payload.message, history, language, response)
    _save_and_log(
        request,
        payload.message,
        response,
        state.corpus_version,
        conversation,
        retrieval.chunks,
    )
    return response


async def _events(
    request: Request,
    message: str,
    state: object,
    conversation: Conversation,
    history: list[Turn],
    started_at: float,
) -> AsyncIterator[bytes]:
    """The SSE body of a ``stream=true`` chat; every failure ends in an ``error`` event."""
    cid = conversation.conversation_id
    language = conversation.language
    yield _sse("meta", {"conversation_id": cid})
    try:
        chunks: list[Chunk] = []
        streamed = False
        if not message.strip():
            response = _response(
                EMPTY_MESSAGE[language],
                cid,
                False,
                [],
                Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
                started_at,
            )
        elif cached := _cached(state, message, history, language, conversation, started_at):
            response = cached
        else:
            rewrite = await _rewrite(state, message, history)
            retrieval = await state.retrieval.retrieve(message, history, rewrite=rewrite.rewrite)
            chunks = retrieval.chunks
            response = None
            answer = _answer_stream(message, retrieval, language, cid, state, started_at, history)
            async for item in answer:
                if isinstance(item, str):
                    streamed = True
                    yield _sse("delta", {"text": item})
                else:
                    response = item
            response = _add_usage(response, rewrite.usage)
            _remember(state, message, history, language, response)
        if not streamed:  # a notice, a fallback or a refusal: shown as one delta
            yield _sse("delta", {"text": response.answer})
        _save_and_log(request, message, response, state.corpus_version, conversation, chunks)
        log.info(
            "chat_stream_completed",
            conversation_id=cid,
            llm_calls=response.usage.llm_calls,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=response.latency_ms,
        )
        yield _sse("done", response.model_dump())
    except Exception as exc:  # the 200 is already sent: the error goes in the stream
        code = getattr(exc, "code", None)
        if not isinstance(code, str):
            log.exception("chat_stream_failed")
            code, text = "internal_error", "The answer could not be completed."
        else:
            text = getattr(exc, "message", str(exc))
        yield _sse("error", {"error": {"code": code, "message": text}})


def _cached(
    state: object,
    message: str,
    history: list[Turn],
    language: QueryLang,
    conversation: Conversation,
    started_at: float,
) -> ChatResponse | None:
    """A cached answer to this first question on this corpus: 0 LLM calls."""
    cache = getattr(state, "answer_cache", None)
    if cache is None or history:
        return None
    hit = cache.get(cache_key(message, language, state.corpus_version))
    if hit is None:
        return None
    answer, sources = hit
    usage = Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0)
    return _response(answer, conversation.conversation_id, True, sources, usage, started_at)


def _remember(
    state: object, message: str, history: list[Turn], language: QueryLang, response: ChatResponse
) -> None:
    """Keep a grounded LLM answer to a first question (never a notice, refusal or fallback)."""
    cache = getattr(state, "answer_cache", None)
    if cache is None or history or not response.grounded or not response.usage.llm_calls:
        return
    cache.put(cache_key(message, language, state.corpus_version), response.answer, response.sources)


def _sse(event: str, data: object) -> bytes:
    return format_sse_event(event=event, data_str=json.dumps(data, ensure_ascii=False))


async def _rewrite(state: object, message: str, history: list[Turn]) -> RewriteOutcome:
    """The optional first LLM call (query-rewrite Req 1); never blocks the answer."""
    rewriter = getattr(state, "rewriter", None)
    if rewriter is None:
        return RewriteOutcome(None, GenerationUsage(llm_calls=0))
    previous = history[-1].user_message if history else None

    def allow() -> bool:  # leave a quota slot for the answer call
        return state.breaker.allow() and state.quota.acquire(keep=1)

    # A provider chain answers the rewrite with its cheaper model and quota.
    for_rewrite = getattr(state.generator, "for_rewrite", None)
    generator = for_rewrite() if for_rewrite else state.generator
    outcome = await rewriter.rewrite(generator, message, previous, allow)
    if outcome.failed:
        state.breaker.record_failure()
    elif outcome.usage.llm_calls:
        state.breaker.record_success()
    if outcome.error:
        log.warning("rewrite_skipped", reason=outcome.error)
    return outcome


def _add_usage(response: ChatResponse, extra: GenerationUsage) -> ChatResponse:
    if not extra.llm_calls:
        return response
    usage = response.usage
    total = Usage(
        llm_calls=usage.llm_calls + extra.llm_calls,
        prompt_tokens=usage.prompt_tokens + extra.prompt_tokens,
        completion_tokens=usage.completion_tokens + extra.completion_tokens,
    )
    return response.model_copy(update={"usage": total})


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
            NO_ANSWER[language],
            conversation_id,
            False,
            [],
            Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0),
            started_at,
        )

    if not state.breaker.allow() or not state.quota.acquire():
        answer, sources, grounded = extractive_fallback(retrieval.chunks, message, language)
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
    return _finish(
        generation, system, message, retrieval, language, conversation_id, state, started_at
    )


async def _answer_stream(
    message: str,
    retrieval: Retrieval,
    language: QueryLang,
    conversation_id: str,
    state: object,
    started_at: float,
    history: list[Turn],
) -> AsyncIterator[str | ChatResponse]:
    """``_answer`` for ``stream=true``: safe text deltas, then the final ``ChatResponse``."""
    stream = getattr(state.generator, "stream", None)
    if retrieval.decision == "refuse" or stream is None:
        yield await _answer(
            message, retrieval, language, conversation_id, state, started_at, history
        )
        return
    if not state.breaker.allow() or not state.quota.acquire():
        answer, sources, grounded = extractive_fallback(retrieval.chunks, message, language)
        usage = Usage(llm_calls=0, prompt_tokens=0, completion_tokens=0)
        yield _response(answer, conversation_id, grounded, sources, usage, started_at)
        return
    system, user = build_prompt(message, language, retrieval.chunks, history=history)
    shown = StreamFilter(retrieval.chunks, system)
    generation = GenerationResult(None, GenerationUsage(llm_calls=1), "invalid_response")
    async for item in stream(system, user):
        if isinstance(item, GenerationResult):
            generation = item
        elif text := shown.feed(item):
            yield text
    if generation.ok and (tail := shown.finish()):
        yield tail
    yield _finish(
        generation, system, message, retrieval, language, conversation_id, state, started_at
    )


def _finish(
    generation: GenerationResult,
    system: str,
    message: str,
    retrieval: Retrieval,
    language: QueryLang,
    conversation_id: str,
    state: object,
    started_at: float,
) -> ChatResponse:
    if generation.ok:
        state.breaker.record_success()
        text = generation.text or ""
        if leaks_prompt(text, system):  # the prompt or the passages, not an answer (I26)
            log.warning("answer_withheld", reason="prompt_leak")
            text = ""
        answer, sources, grounded = build(text, retrieval.chunks, query=message)
        answer = answer or NO_ANSWER[language]  # a bare NOT_FOUND or a withheld reply
    else:
        state.breaker.record_failure()
        answer, sources, grounded = extractive_fallback(retrieval.chunks, message, language)
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
    corpus_version: str,
    conversation: Conversation,
    chunks: list[Chunk],
) -> None:
    source_ids = {source.doc_id for source in response.sources}
    chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.doc_id in source_ids]
    request.app.state.micro_bot_state.conversations.save(
        conversation.conversation_id,
        Turn(user_message=message, assistant_message=response.answer),
        chunk_ids,
    )
    set_request_log_context(
        request,
        llm_calls=response.usage.llm_calls,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        corpus_version=corpus_version,
        conversation_id=response.conversation_id,
        # Both logged so a reply in the wrong language is countable (titles are not counted).
        language=conversation.language,
        answer_language=detect_language(strip_references(response.answer)),
    )


def _query_language(value: str) -> QueryLang:
    if value not in {"ar", "fr", "en"}:
        raise ValueError("language detector must return ar, fr, or en")
    return cast(QueryLang, value)


def _latency_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)
