"""Recording / replay generator, budget and faults (Req 5.1, 5.2, 7.1)."""

import threading
from types import SimpleNamespace

from eval.e2e.generator import (
    Budget,
    FaultyGenerator,
    OllamaClient,
    RecordingGenerator,
    ThreadedClient,
    prompt_key,
)
from micro_bot.llm.client import GenerationResult, GenerationUsage


class SyncClient:
    def __init__(self) -> None:
        self.calls = 0
        self.threads: set[str] = set()

    def generate(self, system, user):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        return GenerationResult(f"answer to {user}", GenerationUsage(1, 10, 5))


def _gen(tmp_path, mode="record", limit=5, client=None):
    inner = ThreadedClient(client or SyncClient())
    return RecordingGenerator(inner, tmp_path, mode, Budget(limit), "m")


async def test_record_then_replay_without_calls(tmp_path):
    client = SyncClient()
    recorder = _gen(tmp_path, client=client)
    live = await recorder.generate("Today is 2026-10-11. sys", "q1")
    assert live.text == "answer to q1" and client.calls == 1 and recorder.calls == 1
    assert threading.main_thread().name not in client.threads
    replay = RecordingGenerator(None, tmp_path, "replay", Budget(0), "m")
    again = await replay.generate("Today is 2026-12-31. sys", "q1")
    assert again.text == live.text and again.usage.llm_calls == 1 and replay.calls == 0


async def test_replay_miss_when_the_prompt_changed(tmp_path):
    await _gen(tmp_path).generate("sys", "q1")
    miss = await RecordingGenerator(None, tmp_path, "replay", Budget(0), "m").generate("sys2", "q1")
    assert miss.error == "replay_miss" and miss.usage.llm_calls == 0


async def test_budget_stops_provider_calls(tmp_path):
    client = SyncClient()
    gen = _gen(tmp_path, mode="live", limit=2, client=client)
    results = [await gen.generate("s", f"q{i}") for i in range(4)]
    assert client.calls == 2
    assert [r.error for r in results] == [None, None, "eval_budget", "eval_budget"]
    assert results[-1].usage.llm_calls == 0
    assert not list(tmp_path.iterdir())  # live mode records nothing


async def test_not_configured_without_a_client(tmp_path):
    result = await RecordingGenerator(None, tmp_path, "live", Budget(5), "m").generate("s", "q")
    assert result.error == "not_configured"


def test_key_masks_the_date_only():
    assert prompt_key("m", "Today is 2026-10-11. x", "q") == prompt_key(
        "m", "Today is 2027-01-01. x", "q"
    )
    assert prompt_key("m", "x", "q") != prompt_key("other", "x", "q")


async def test_faults_spend_one_call():
    for kind, error in [
        ("timeout", "timeout"),
        ("rate_limited", "rate_limited"),
        ("provider_5xx", "provider_5xx"),
        ("empty", "invalid_response"),
    ]:
        result = await FaultyGenerator(kind).generate("s", "q")
        assert not result.ok and result.error == error and result.usage.llm_calls == 1


class FakeOpenAI:
    """Just enough of the OpenAI SDK: chat.completions.create(**kwargs)."""

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.kwargs: dict = {}
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _completion(text, prompt=7, completion=3):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def test_ollama_client_makes_one_deterministic_call():
    fake = FakeOpenAI(_completion("Bonjour [doc]"))
    result = OllamaClient("gemma4:e4b", client=fake).generate("sys", "q")
    assert result.ok and result.text == "Bonjour [doc]"
    assert (result.usage.llm_calls, result.usage.prompt_tokens) == (1, 7)
    assert fake.kwargs["model"] == "gemma4:e4b" and fake.kwargs["temperature"] == 0
    assert [m["role"] for m in fake.kwargs["messages"]] == ["system", "user"]
    assert fake.kwargs["extra_body"] == {"reasoning_effort": "none"}  # no thinking pass


def test_ollama_client_failures_are_results():
    result = OllamaClient("m", client=FakeOpenAI(TimeoutError("slow"))).generate("s", "q")
    assert not result.ok and result.usage.llm_calls == 1 and result.error
    empty = OllamaClient("m", client=FakeOpenAI(_completion("  "))).generate("s", "q")
    assert empty.error == "invalid_response"


async def test_citing_stub_cites_the_first_passage_label():
    from eval.e2e.generator import CitingStub
    from micro_bot.llm.prompts import build_prompt
    from micro_bot.schemas import Chunk

    chunk = Chunk(chunk_id="c", doc_id="d", title="t", text="x", score=1.0)
    system, user = build_prompt("q", "en", [chunk])
    assert (await CitingStub().generate(system, user)).text == "Stub answer [S1]."
    system, user = build_prompt("q", "en", [])
    assert (await CitingStub().generate(system, user)).text == "NOT_FOUND I don't know."
