# Design Document

## Overview

```
cases.jsonl ─┐                       ┌─ http driver ──────► running app (compose / uvicorn)
perturb.py ──┼─► runner (suite) ─────┤
corpus/ ─────┘   budget, pacing      └─ inprocess driver ─► create_app(state=...) via ASGITransport
                       │                   real ingest + retrieval, RecordingGenerator, faults
                       ▼
               score.py (rules) ─► judge.py (optional) ─► report.py ─► thresholds.json ─► exit code
```

One run = one suite on one driver. Every response goes through the same scorers whatever the
driver, so a live run and an in-process run are comparable.

### What the code does today (checked 2026-10-11, `origin/dev` f69ee94)

| Fact | Where | Consequence for the harness |
|---|---|---|
| `default_state()` leaves `generator` at `FakeGenerationService` (`error="disabled"`) | `main.py:304` | Live app never calls the LLM; every answer is `extractive_fallback`. Path check (Req 3.5) reports it; generation is "not evaluated" until fixed. |
| `OpenRouterClient.generate` is sync, `GenerationService.generate` is async | `llm/client.py`, `main.py:69` | In-process driver wraps it with `asyncio.to_thread` in its own `RecordingGenerator`; the app fix is eequaled's. |
| Gate refusal text is always `"I don't know based on the available documents."` | `api/chat.py:113` | Separate `refusal_language` metric, else every ar/fr refusal fails the language check. |
| Quota or breaker closed -> `extractive_fallback`, `grounded` may be true, `llm_calls=0` | `api/chat.py:121` | Classified `fallback`, never counted as an LLM answer. |
| LLM failure -> fallback with `llm_calls=1` | `api/chat.py:139` | Fault suite expects 200 + `llm_calls == 1`. |
| Cache skipped when the request has a `conversation_id`; unknown id -> new conversation, no history | `api/chat.py:43-57`, `conversation.py` | `eval-<uuid>` per request bypasses the cache without an API change. |
| Rate limit 20 requests / 60 s per client IP | `config.py`, `rate_limit.py` | HTTP driver paces; in-process driver passes its own `SlidingWindowRateLimiter` with a high limit. |
| `grounded` needs at least one valid `[doc_id]` citation and no refusal phrase | `answer.py` | Answer pass requires `grounded`; evidence checked in `sources[].snippet` (first 500 chars of the chunk). |
| Quota state lives in `DATA_DIR/llm_quota.json` | `llm/quota.py` | In-process runs use a temp `DATA_DIR`, so they do not eat the app's quota, but the provider's daily limit is shared: `--max-llm-calls` still applies. |

## Layout

```
eval/e2e/
├── __init__.py
├── cases.py        Case model (pydantic, extra="forbid"), load + validate, dev/test split
├── convert.py      questions.jsonl -> cases_retrieval.jsonl (generated, checked by a test)
├── cases.jsonl     hand-written base cases (Req 1); ids unique across both files
├── corpus/         adversarial + canary documents, manifest.json (Req 2)
├── perturb.py      deterministic variants (Req 6.1)
├── drivers.py      HttpDriver, InProcessDriver: same async interface
├── generator.py    RecordingGenerator (record / replay), FaultyGenerator
├── score.py        rule checks -> CheckResult(name, status, reason)
├── judge.py        optional judge client + verdict cache
├── suites.py       quality, robustness, contract, faults, cache, load, restart
├── report.py       summary, markdown, scorecard
├── compare.py      diff two result files
├── thresholds.json
├── recordings/     replay store (committed: makes CI replay possible)
└── run.py          CLI: python -m eval.e2e.run --suite ... --target ...
```

Results go to `eval/results/e2e/`. Nothing in `src/` changes.

## Components

### Case (`cases.py`)

```python
class Case(BaseModel, extra="forbid", frozen=True):
    id: str
    question: str
    lang: Literal["ar", "fr", "en"]
    role: Role
    expect: Literal["answer", "refuse"]
    docs: list[str] = []  # every topic that holds the answer
    gold: list[str] = []
    evidence: list[str] = []
    must_not: list[str] = []
    history: list[TurnIn] = []
    slices: list[str] = []
```

`split(case) = "test" if int(sha1(id)[:8], 16) % 5 == 0 else "dev"`. Variants inherit the
split of their base so a base and its variants never straddle the split. Public cases get every
canary appended to `must_not` at load time (from the corpus manifest), and every case gets the
injection markers, so the case file does not repeat them.

### Drivers (`drivers.py`)

```python
class Driver(Protocol):
    async def setup(self, corpus: Corpus) -> dict[str, str]: ...  # topic -> doc_id
    async def chat(self, case: Case) -> Observation: ...
    async def retrieve(self, case: Case) -> Retrieval: ...  # inprocess only
    async def request(self, method, path, **kw) -> httpx.Response: ...  # contract suite
    async def teardown(self) -> None: ...
```

`Observation(status, body, latency_ms_client, error)`. Follow-up cases need their history in
the server's conversation. Sending the earlier turns as real requests would cost LLM calls, so
the in-process driver seeds the `ConversationStore` directly (`create(lang)` then `save(id,
turn, [])` per turn) and sends that `conversation_id`. The HTTP driver cannot seed, so it marks
follow-up cases `skip` unless `--followups` is given (then it spends the calls).

- **HttpDriver:** `httpx.AsyncClient(base_url, timeout=60)`; headers from env; a token bucket
  under `CHAT_RATE_LIMIT_REQUESTS / CHAT_RATE_LIMIT_WINDOW_S` (read from env, defaults 20/60).
  `setup` uploads each corpus file with `topic="eval-<topic>"`; an upload that returns
  `duplicate` is fine (idempotent runs).
- **InProcessDriver:** temp `DATA_DIR`; `IngestPipeline` as in `run_eval._run` (real encoders,
  `AsyncQdrantClient(":memory:")` or `--qdrant-url`); a small `DocumentBackend` adapter over that
  pipeline instance (the real one reads the module-level pipeline); `ApplicationState(...,
  retrieval=HybridRetriever.from_pipeline(...), generator=RecordingGenerator(...),
  rate_limiter=SlidingWindowRateLimiter(10_000, 60), language_detector=RealLanguageDetector())`;
  `httpx.AsyncClient(transport=httpx.ASGITransport(app))`. `retrieve()` calls the retriever
  directly for the retrieval layer.

### Generators (`generator.py`)

`RecordingGenerator(inner: OpenRouterClient | None, store: Path, mode: "live"|"record"|"replay",
budget: Budget)`. Key = sha256 of `model + "\n" + system + "\n" + user` with the date line of
the system prompt (`Today is YYYY-MM-DD.`) masked, else every recording expires at midnight.
`replay` on a missing key returns `GenerationResult(None, usage(0), "replay_miss")` and the case
fails with that reason. `live` calls `asyncio.to_thread(inner.generate, ...)` only while
`budget.take()` succeeds, else returns `error="eval_budget"` with `llm_calls=0`.

`FaultyGenerator(kind)` returns the `GenerationResult` the real client returns for `timeout`,
`rate_limited`, `provider_5xx`, `invalid_response`, all with `llm_calls=1`. Retrieval faults wrap
the retriever and raise `IndexUnavailable` / `RerankError` from `ingest.errors`.

### Scoring (`score.py`)

Pure functions `(case, observation, corpus) -> list[CheckResult]`: `contract`, `llm_calls`,
`behaviour`, `gold`, `evidence`, `must_not`, `access`, `language`, `refusal_language`, `path`.
Matching uses `fold_for_search` on both sides (same folding as the retrieval eval labels), so
diacritics, case and Arabic letter variants do not cause false failures. A case passes when no
check is `fail`. Path classification:

| `llm_calls` | `grounded` | sources | path |
|---|---|---|---|
| 1 | any | any | `llm` |
| 0 | false | none, answer = refusal text | `gate_refusal` |
| 0 | any | some, or any answer text | `fallback` |
| 0 | any | any, response equals a cached one (cache suite only) | `cache` |

Retrieval layer (`--layer retrieval`): `behaviour` from `Retrieval.decision`, `evidence` in any
returned chunk text, `must_not` / `access` on chunk texts and doc ids. No LLM involved.

### Judge (`judge.py`)

`openai.OpenAI(base_url=EVAL_JUDGE_BASE_URL, api_key=EVAL_JUDGE_API_KEY or "none",
max_retries=0, timeout=60)`, `temperature=0`, rubric in the module with `RUBRIC_VERSION`.
Prompt: answer + cited snippets, both fenced as untrusted; asks for JSON only. Verdicts cached
in `eval/e2e/judge_cache.jsonl`. Not committed when it holds answers from a private corpus.

### Perturbations (`perturb.py`)

`random.Random(f"{seed}:{case.id}:{kind}")` per variant. Kinds: `typo` (swap two adjacent
letters inside words longer than 3, once per 8 letters, never inside digits or `#hex`),
`diacritics` (Arabic: add a fatha/kasra to 30% of letters or strip all harakat and tatweel;
Latin: strip accents), `noise` (random case on Latin letters, doubled spaces, trailing `??`),
`filler` (prefix "please tell me" / "dis-moi" / "من فضلك أخبرني" by `lang`). Gold, evidence and
expectations are unchanged.

### Suites (`suites.py`)

| Suite | Driver | LLM calls | What it runs |
|---|---|---|---|
| `quality` | both | up to `--max-llm-calls` | all cases of the chosen split |
| `robustness` | inprocess | 0 (retrieval layer) | base + variants, `--repeat` |
| `contract` | both | 0 | auth matrix, 422s, error shape |
| `faults` | inprocess | 0 (fake generator) | Req 7.1 table |
| `cache` | both | <= 1 | same question twice, then a corpus change |
| `load` | both | 0 | refuse-expected + gate-path cases, C users, S seconds |
| `restart` | http + `--allow-docker` | 0 | `docker compose restart app`, then `qdrant`, then re-check |

`all` = `contract, faults, robustness, quality (replay), cache, load` on inprocess.

### Report (`report.py`)

Summary groups results by slice, `lang`, `role`, `path`. Rates are over cases with a `pass` or
`fail` status (skips excluded, their count shown). Scorecard rows: `metric, value, threshold,
ok`. Exit code 1 when any row fails, 2 when the run itself crashed.

## Error handling

- A driver exception on one case becomes `Observation(error=...)` and a `contract` failure for
  that case; the run continues. Setup failures (corpus upload, Qdrant down) abort with exit 2.
- Ctrl-C writes the partial result with `"interrupted": true`.

## Testing strategy

- Unit (default CI): case loading and validation errors; split stability; each perturbation on
  ar/fr/en strings with a fixed seed (golden outputs); every scorer on hand-built observations;
  path classification table; budget stop; recorder record/replay/miss and date masking;
  compare; scorecard exit codes.
- Driver test: `InProcessDriver` with `tests/pipeline_helpers.py` fakes (fake encoders, fake
  reranker, `FakeGenerationService`) on 2 tiny documents, one members-only with a canary: the
  public case must not leak, the faults table must hold. No models needed.
- Real runs are manual (`--target inprocess` with models, or `--target http` on compose).
