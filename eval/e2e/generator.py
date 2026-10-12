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


OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaClient:
    """A local model through Ollama's OpenAI-compatible API: a test stand-in, not the provider.

    Same contract as ``OpenRouterClient.generate`` (one call, failures as results, same parsing
    and error kinds), so the answer layer sees what it would see from OpenRouter. Temperature 0
    with a fixed seed keeps reruns comparable. Its citations (``[S1]``, ``[doc_id: X]``, ...) go
    to ``answer.build`` unchanged, as a provider's would.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 110,  # under the in-process driver's 120 s request timeout
        client: object | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key="ollama", base_url=base_url, timeout=timeout, max_retries=0)
        self._client = client

    def generate(self, system: str, user: str) -> GenerationResult:
        from mcclub_rag.llm.client import OpenRouterClient

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                seed=0,
                timeout=self.timeout,
            )
        except Exception as exc:  # same as the real client: failures are results
            return GenerationResult(None, GenerationUsage(1), OpenRouterClient._error_kind(exc))
        return OpenRouterClient._parse_completion(completion, calls=1)


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
    """Answers by citing the first passage's label, one 'call' each, no network.

    For the cache and load suites, which test the system around generation, not generation.
    """

    _FIRST_PASSAGE = "[S1]\n```untrusted-source"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system: str, user: str) -> GenerationResult:
        self.calls += 1
        text = "Stub answer [S1]." if self._FIRST_PASSAGE in system else "I don't know."
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
