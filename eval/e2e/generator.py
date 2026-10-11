"""Generation services for the in-process driver (Req 5.2, 7.1).

``RecordingGenerator`` sits behind the app's async ``GenerationService`` protocol in front of
the real ``OpenRouterClient`` (sync, so through ``ThreadedClient``). It spends at most
``Budget`` provider calls and records or replays generations by prompt hash. The system
prompt's date line is masked in the key, otherwise every recording would expire at midnight.
``FaultyGenerator`` returns what the real client returns for each provider failure, so the
fault suite needs no network.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Protocol

from mcclub_rag.llm.client import GenerationResult, GenerationUsage

Mode = Literal["live", "record", "replay"]
_DATE = re.compile(r"Today is \d{4}-\d{2}-\d{2}\.")


class _SyncClient(Protocol):
    def generate(self, system: str, user: str) -> GenerationResult: ...


class _Inner(Protocol):
    async def generate(self, system: str, user: str) -> GenerationResult: ...


class Budget:
    """At most ``limit`` provider calls for the whole run (the app has 50 a day)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def prompt_key(model: str, system: str, user: str) -> str:
    masked = _DATE.sub("Today is <date>.", system)
    return hashlib.sha256(f"{model}\n{masked}\n{user}".encode()).hexdigest()


class RecordingGenerator:
    def __init__(
        self, inner: _Inner | None, store: Path, mode: Mode, budget: Budget, model: str
    ) -> None:
        self.inner = inner
        self.store = store
        self.mode = mode
        self.budget = budget
        self.model = model
        self.calls = 0  # provider calls actually made

    async def generate(self, system: str, user: str) -> GenerationResult:
        key = prompt_key(self.model, system, user)
        path = self.store / f"{key}.json"
        if self.mode == "replay":
            if not path.is_file():
                return GenerationResult(None, GenerationUsage(llm_calls=0), "replay_miss")
            return _load(path)
        if self.inner is None:
            return GenerationResult(None, GenerationUsage(llm_calls=0), "not_configured")
        if not self.budget.take():
            return GenerationResult(None, GenerationUsage(llm_calls=0), "eval_budget")
        result = await self.inner.generate(system, user)
        self.calls += result.usage.llm_calls
        if self.mode == "record" and result.ok:
            self.store.mkdir(parents=True, exist_ok=True)
            _save(path, result)
        return result


class ThreadedClient:
    """Run a sync client's ``generate`` in a worker thread (the app's protocol is async)."""

    def __init__(self, client: _SyncClient) -> None:
        self.client = client

    async def generate(self, system: str, user: str) -> GenerationResult:
        return await asyncio.to_thread(self.client.generate, system, user)


FAULTS = {
    "timeout": "timeout",
    "rate_limited": "rate_limited",
    "provider_5xx": "provider_5xx",
    "empty": "invalid_response",
}


class FaultyGenerator:
    """Always fails like the real client does for ``kind`` (one provider call spent)."""

    def __init__(self, kind: str) -> None:
        self.error = FAULTS[kind]

    async def generate(self, system: str, user: str) -> GenerationResult:
        return GenerationResult(None, GenerationUsage(llm_calls=1), self.error)


class CitingStub:
    """Answers by citing the first passage's doc_id, one 'call' each, no network.

    For the cache and load suites, which test the system around generation, not generation.
    """

    _DOC_ID = re.compile(r"^doc_id: (\S+)$", re.MULTILINE)

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        match = self._DOC_ID.search(system)
        text = f"Stub answer [{match.group(1)}]." if match else "I don't know."
        return GenerationResult(text, GenerationUsage(1, 0, 0))


def _save(path: Path, result: GenerationResult) -> None:
    usage = result.usage
    data = {
        "text": result.text,
        "usage": [usage.llm_calls, usage.prompt_tokens, usage.completion_tokens],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load(path: Path) -> GenerationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    calls, prompt, completion = data["usage"]
    # Replayed as an LLM answer (so it scores like the live one); it costs no provider call,
    # which RecordingGenerator.calls counts separately.
    return GenerationResult(data["text"], GenerationUsage(1, prompt, completion))
