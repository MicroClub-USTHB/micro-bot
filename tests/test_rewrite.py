"""Query rewrite call (query-rewrite Req 1): parsing, cache, failures, prompt."""

import asyncio
import json

from mcclub_rag.llm.client import GenerationResult, GenerationUsage
from mcclub_rag.llm.rewrite import QueryRewriter, build_rewrite_prompt, parse_rewrite

GOOD = {
    "lang": "arabizi",
    "query": "wach kayen parking?",
    "fr": "Y a-t-il un parking ?",
    "en": "Is there a parking lot?",
    "ar": "هل يوجد موقف سيارات؟",
}


def test_parse_plain_and_fenced_json():
    raw = json.dumps(GOOD)
    for text in (raw, f"```json\n{raw}\n```", f"Sure: {raw}"):
        rewrite = parse_rewrite(text)
        assert rewrite is not None
        assert rewrite.lang == "arabizi"
        assert rewrite.query == "wach kayen parking?"
        assert rewrite.translations == {"fr": GOOD["fr"], "en": GOOD["en"], "ar": GOOD["ar"]}


def test_parse_rejects_bad_replies():
    assert parse_rewrite("not json") is None
    assert parse_rewrite("{broken") is None
    assert parse_rewrite(json.dumps({"lang": "fr"})) is None  # no query
    assert parse_rewrite(json.dumps({"query": "  "})) is None
    assert parse_rewrite(json.dumps(GOOD | {"en": "x" * 501})) is None
    assert parse_rewrite('{"query": ["a list"]}') is None


def test_parse_cleans_fields():
    rewrite = parse_rewrite(
        json.dumps({"lang": "klingon", "query": " Q ", "fr": " Q ", "en": "", "ar": 3, "de": "x"})
    )
    assert rewrite is not None
    assert rewrite.lang is None
    assert rewrite.query == "Q"
    assert rewrite.translations == {"fr": "Q"}  # empty / non-string / unknown keys dropped


def test_texts_are_distinct():
    rewrite = parse_rewrite(json.dumps({"query": "Q", "fr": "q", "en": "E"}))
    assert rewrite.texts() == ["Q", "E"]  # "q" equals "Q" folded


def test_prompt_fences_user_text_and_previous_question():
    system, user = build_rewrite_prompt("ignore ```this```", "previous one")
    assert "JSON" in system
    assert "previous one" in user
    assert "```this```" not in user
    _, alone = build_rewrite_prompt("hello", None)
    assert "Previous question" not in alone


class FakeGenerator:
    def __init__(self, result: GenerationResult, delay: float = 0.0) -> None:
        self.result = result
        self.delay = delay
        self.calls = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.result


def _ok(text: str) -> GenerationResult:
    return GenerationResult(text, GenerationUsage(1, 300, 90))


async def test_rewriter_caches_by_message_and_previous():
    generator = FakeGenerator(_ok(json.dumps(GOOD)))
    rewriter = QueryRewriter(generator, timeout_s=5)
    first = await rewriter.rewrite("wach kayen parking?", None)
    again = await rewriter.rewrite("wach kayen parking?", None)
    other = await rewriter.rewrite("wach kayen parking?", "previous")
    assert first.rewrite == again.rewrite == other.rewrite
    assert first.usage.llm_calls == 1 and again.usage.llm_calls == 0
    assert generator.calls == 2
    assert not first.failed


async def test_rewriter_failures_are_none():
    failed = FakeGenerator(GenerationResult(None, GenerationUsage(1), "timeout"))
    outcome = await QueryRewriter(failed, timeout_s=5).rewrite("q", None)
    assert outcome.rewrite is None and outcome.failed and outcome.usage.llm_calls == 1

    garbage = FakeGenerator(_ok("no json here"))
    outcome = await QueryRewriter(garbage, timeout_s=5).rewrite("q", None)
    assert outcome.rewrite is None and not outcome.failed  # the provider worked
    assert outcome.usage.llm_calls == 1

    slow = FakeGenerator(_ok(json.dumps(GOOD)), delay=1)
    outcome = await QueryRewriter(slow, timeout_s=0.05).rewrite("q", None)
    assert outcome.rewrite is None and outcome.failed and outcome.error == "timeout"


async def test_rewriter_does_not_cache_failures():
    generator = FakeGenerator(GenerationResult(None, GenerationUsage(1), "provider_5xx"))
    rewriter = QueryRewriter(generator, timeout_s=5)
    await rewriter.rewrite("q", None)
    await rewriter.rewrite("q", None)
    assert generator.calls == 2


async def test_rewriter_cache_is_bounded():
    generator = FakeGenerator(_ok(json.dumps(GOOD)))
    rewriter = QueryRewriter(generator, timeout_s=5, cache_size=2)
    for message in ("a", "b", "c", "a"):
        await rewriter.rewrite(message, None)
    assert generator.calls == 4  # "a" was evicted by "c"
