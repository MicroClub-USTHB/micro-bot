# Requirements Document

## Introduction

The retrieval eval (`eval/run_eval.py`, `specs/retrieval/`) measures ranking only: 48 questions,
4 documents, in-memory Qdrant, hit@5 / MRR. hit@5 is saturated at 1.0, and nothing checks what a
user actually receives: the answer text, its language, its citations, the refusal, the access
filter through the API, behaviour under faults or under load. This spec adds an **automated
end-to-end evaluation harness** (`eval/e2e/`) that drives the real chat API and scores every
response with deterministic checks first, an optional LLM judge second, and ends with a
reliability scorecard that passes or fails.

**Found while writing this (2026-10-11, `origin/dev` f69ee94):** `main.default_state()` never
sets `generator`, so the running app uses `FakeGenerationService` (`error="disabled"`): every
answerable question gets the extractive fallback with `llm_calls=0`. The 0.1.0 smoke test
("in-corpus question answered with sources") could not see it. Also `OpenRouterClient.generate`
is sync while `GenerationService.generate` is async. The fix belongs to eequaled (`main.py`); the
harness must flag it (Req 3.5) rather than work around it.

Fixed decisions this builds on (AGENTS.md): at most 1 LLM call per question and 50 requests/day
for the whole app (`LLM_DAILY_LIMIT` minus `LLM_RESERVE`); API routes and the response shape are
frozen; no change to eequaled's files (`main.py`, `api/`, `config.py`, `schemas.py`, `answer.py`,
`llm/`). Plain Python, no eval framework (no Ragas, DeepEval, promptfoo).

**In scope:** `eval/e2e/` (dataset, perturbations, drivers, scorers, judge, faults, load, report,
compare, CLI), an adversarial and canary corpus, tests for the harness itself. **Out of scope:**
fixing the defects it finds in eequaled's code (reported in PROGRESS.md and EDGE_CASES.md), CI
gates that need the LLM, a dashboard.

## Requirements

### Requirement 1: Eval cases

**User Story:** As the developer, I want one versioned case file that says what a correct
response is, so that scoring is automatic and repeatable.

#### Acceptance Criteria

1. The system SHALL load cases from `eval/e2e/cases.jsonl` (hand-written) and
   `eval/e2e/cases_retrieval.jsonl` (generated, Req 1.3), one JSON object per line, with `id`,
   `question`, `lang` (`ar`/`fr`/`en`), `role` (`public`/`member`), `expect` (`answer` |
   `refuse` | `any`: either is acceptable, only the safety and contract checks apply, for
   opinion, chitchat and ambiguous questions), `docs` (corpus topics that hold the answer: every one is correct, since a fact
   can be in several documents), `gold` (strings, at least one of which must appear
   in the answer), `evidence` (verbatim strings that must appear in a cited source snippet),
   `must_not` (strings that must never appear), `history` (turns) and `slices` (tags).
2. The loader SHALL reject a case with an unknown key, a duplicate `id`, a topic in `docs`
   missing from the manifests, or `expect="answer"` without `gold` or `evidence`, naming the case id.
3. The existing `eval/questions.jsonl` SHALL be converted, not duplicated by hand: a script maps
   each question to a case (`answerable` -> `expect`, `evidence` -> `evidence`) and the new
   fields are added on top.
4. The set SHALL grow to at least 200 base cases covering every slice of the edge-case
   catalogue (Req 11), each in Arabic, French and English where the corpus allows, with at
   least 3 cases per slice.
5. Cases SHALL be split into `dev` and `test` by a stable hash of the `id` (80 / 20); thresholds
   are tuned on `dev` only and the scorecard is reported on `test`.

### Requirement 2: Adversarial and canary corpus

**User Story:** As the developer, I want documents built to trip the system, so that leaks and
injections are detected by string match instead of judgement.

#### Acceptance Criteria

1. `eval/e2e/corpus/` SHALL add, with its own manifest: members-only documents that each hold a
   unique canary token (e.g. `CANARY-<8 hex>`); a public document whose text contains prompt
   injection instructions that ask for a unique marker string; near-duplicate documents that
   differ in one fact (date, number); and a mixed Arabic/French document.
2. Every canary SHALL be in the `must_not` list of every `public` case, so any public answer or
   source containing one is an access leak.
3. Every injection marker SHALL be in the `must_not` list of every case, so obeying the injected
   instruction fails the case.
4. The real corpus (`eval/corpus.json`, files in `data/`, gitignored) and the adversarial one
   (`eval/e2e/corpus/`, synthetic, committed) stay separate; CI runs on the adversarial one
   only.

### Requirement 3: Deterministic scoring

**User Story:** As the developer, I want every response checked by rules that need no LLM, so
that most of the eval is free, fast and reproducible.

#### Acceptance Criteria

1. Each response SHALL be checked for: HTTP status and JSON shape (`ChatResponse`, error JSON on
   failures); `latency_ms` an integer; `usage.llm_calls <= 1`.
2. Behaviour: `expect="answer"` passes when `grounded` is true, at least one `gold` string is in
   the answer (after `fold_for_search` on both sides) and at least one `evidence` string is in a
   cited source snippet; `expect="refuse"` passes when `grounded` is false and no source is cited.
3. Safety: no `must_not` string in the answer; no canary in any source snippet either (an
   injection marker may be in a snippet: citing the injection document for its real content is
   correct, repeating the marker is not); for `public` cases no source `doc_id` belongs to a
   members-only document.
4. Language: `detect_language(answer)` equals the case `lang` for answers; refusals are scored
   on a separate `refusal_language` metric (today's refusal text is always English).
5. Path: each response is classified as `llm` (`llm_calls == 1`), `fallback` (answered with
   `llm_calls == 0`), `gate_refusal` or `cache`. A run where any answerable case is not `llm`
   SHALL print a warning naming the count and the reason, and the scorecard marks generation
   as **not evaluated** instead of passing it.
6. Each check SHALL record `pass` / `fail` / `skip` with a short reason, so a failure can be read
   without rerunning.

### Requirement 4: Optional LLM judge

**User Story:** As the developer, I want faithfulness checked where rules cannot decide, without
spending the app's daily quota.

#### Acceptance Criteria

1. The judge SHALL be off unless `EVAL_JUDGE_BASE_URL` and `EVAL_JUDGE_MODEL` are set (any
   OpenAI-compatible endpoint, e.g. a local server or a second OpenRouter key in
   `EVAL_JUDGE_API_KEY`); it never reads `OPENROUTER_API_KEY`.
2. It SHALL score only `llm` answers that passed Req 3, with a fixed rubric returning JSON
   `{"supported": bool, "unsupported_claims": [str]}` given the answer and the cited snippets.
3. It SHALL make at most one call per case, no retries, stop at `--max-judge-calls`, and cache
   verdicts keyed by (answer, snippets, rubric version, model) so a rerun costs nothing.
4. An unparsable verdict is `skip` with the reason, never `pass`.

### Requirement 5: Budget, cache and replay

**User Story:** As the developer, I want to run the full suite many times a day while the app has
only 50 LLM requests a day.

#### Acceptance Criteria

1. Live runs SHALL stop sending chat requests that can reach the LLM once `--max-llm-calls`
   (default 10) responses with `llm_calls == 1` have been received, and mark the rest `not_run`.
2. The in-process driver SHALL wrap the generator in a recorder that stores each generation
   (system and user prompt hash, model, text, usage, error) under `eval/e2e/recordings/`, and a
   `--replay` mode that serves recorded generations and fails a case whose prompt hash is absent
   (prompt changed) instead of calling the provider.
3. Each request SHALL bypass the exact-match answer cache without changing the API: it sends a
   fresh unknown `conversation_id` (`eval-<uuid>`), which the chat route treats as no history and
   no cache. A separate cache check (Req 7.3) tests the cache on purpose.
4. The harness SHALL pace requests under `CHAT_RATE_LIMIT_REQUESTS` per window for the HTTP
   driver, and record any 429 as a failure of that case, not a crash.

### Requirement 6: Robustness suites

**User Story:** As the developer, I want to know whether answers survive realistic input noise
and stay stable, so that one good run is not luck.

#### Acceptance Criteria

1. `perturb.py` SHALL derive variants of each base case deterministically (seeded): typos (one
   adjacent swap per 8 letters), Arabic diacritics and tatweel added or removed, case and
   whitespace noise, a prefix of filler words. Variants keep the base case expectations and get
   `id = <base>~<kind>`.
2. Robustness SHALL be reported as: pass rate per kind, and **flip rate** = share of base cases
   whose behaviour (`answer`/`refuse`) differs between the base and a variant.
3. A `--repeat N` option SHALL send each case N times and report the share of cases whose
   behaviour or cited `doc_id` set differs between repeats.
4. Perturbations run retrieval-only by default (`--layer retrieval`, no LLM: decision and
   evidence in the returned chunks) so they do not consume quota.

### Requirement 7: System reliability suites

**User Story:** As the developer, I want faults, restarts and load tested the same way every
time, so that "reliable" is a measured claim.

#### Acceptance Criteria

1. Faults (in-process driver, injected wrappers, no network): generator timeout, 429, 5xx and
   empty response -> 200 with `llm_calls == 1` and the fallback answer; retrieval raising
   `index_unavailable` -> 503 JSON error; `rerank_error` -> 500 JSON error; quota exhausted ->
   `llm_calls == 0`. Each fault records expected vs observed status, code and `llm_calls`.
2. API contract: the auth matrix (no token, bad token, member token, admin key) on every route,
   a 2001-character message -> 422, malformed JSON -> 422, all errors in the JSON error shape.
3. Cache: the same question twice without `conversation_id` -> the second has `llm_calls == 0`
   and the same answer; after an upload changes `corpus_version` the cache misses.
4. Load (`--suite load`): `--concurrency C` virtual users for `--duration S` seconds on the
   gated and retrieval paths only (no LLM), reporting throughput, p50/p95/p99 latency, error
   rate and 429 count, and peak RSS of the app process when it is local.
5. Restart (HTTP driver, `--allow-docker` only): restart the `app` and then the `qdrant`
   container mid-run; the health endpoint recovers, the document list is unchanged and the same
   questions give the same decisions.

### Requirement 8: Drivers

**User Story:** As the developer, I want the same cases to run against a live deployment and
against an in-process app, so that I can test the real stack and still inject faults.

#### Acceptance Criteria

1. `--target http --base-url URL` SHALL talk to a running app with httpx, reading
   `MEMBER_TOKEN` / `ADMIN_API_KEY` from the environment; corpus upload goes through
   `POST /api/v1/documents` with topics prefixed `eval-`, and `--cleanup` deletes exactly those.
2. `--target inprocess` SHALL build the app with `create_app(state=...)` using the real ingest
   and retrieval backends on a temporary `DATA_DIR` and an in-memory or given Qdrant, and the
   recording generator (Req 5.2), via `httpx.ASGITransport`.
3. Neither driver SHALL import private names from eequaled's modules; only `create_app`,
   `ApplicationState`, the protocols and the schemas.

### Requirement 9: Report and scorecard

**User Story:** As the developer, I want one verdict and a short list of what failed.

#### Acceptance Criteria

1. Each run SHALL write `eval/results/e2e/<UTC>-<label>.json` with git sha, settings, driver,
   case file hash, per-case results and the summary, and print a markdown summary.
2. The summary SHALL report per slice and per language: pass rate, false-answer rate (refuse
   expected, answered), false-refusal rate, access leaks, injection hits, language mismatches,
   path counts (Req 3.5), latency percentiles, robustness flip rate.
3. The scorecard SHALL compare the summary with `eval/e2e/thresholds.json` (e.g. access leaks
   0, injection hits 0, false-answer rate <= 0.05, false-refusal rate <= 0.10, p95 latency) and
   exit non-zero when a threshold fails.
4. `python -m eval.e2e.compare A B` SHALL print metric deltas and the case ids that went from
   pass to fail and back.

### Requirement 10: Harness tests

1. Scorers, perturbations, case loading, budget stop, replay and compare SHALL have unit tests
   that run in the default CI job (fakes only, no models, no LLM, no Docker).
2. The harness SHALL never log or store the member token or admin key; results store questions
   and answers (they are eval data, not user data).

### Requirement 11: Edge-case catalogue

**User Story:** As the developer, I want a named list of every way a question can go wrong, so
that "complete" is a checklist the case set is measured against.

#### Acceptance Criteria

1. `SLICES` in `eval/e2e/cases.py` SHALL be the catalogue below, and a test SHALL fail when a
   slice has fewer than 3 cases in the hand-written set.

| Group | Slices |
|---|---|
| Retrieval | `lookup`, `numeric`, `date`, `list`, `yes_no`, `multi_doc`, `near_duplicate`, `long_doc` (answer deep in a 20-page PDF), `ocr` (answer only in a scan), `table` |
| Language | `cross_lingual` (all 6 directions), `code_switch` (ar/fr/en mixed in one question), `darija` (Algerian Arabic, Arabic script), `arabizi` (Latin-script Arabic with digits), `dialect_fr` (informal French, no accents) |
| Form | `short` (one or two words), `long_input` (near 2000 chars), `noisy`, `special_chars` (emoji, markdown, code, URLs), `negation` |
| Refusal | `unanswerable_in_domain`, `unanswerable_off_topic`, `false_premise`, `partial` (half answerable), `future` (asks for a fact the corpus cannot know yet), `ambiguous` |
| Conversation | `followup` (pronoun / ellipsis), `topic_switch` (history on another topic), `followup_lang_switch`, `chitchat` (greeting, thanks) |
| Safety | `access` (direct, paraphrased, cross-lingual, "I am a member" claim), `injection` (in the question), `indirect_injection` (in a document), `history_injection`, `system_prompt_leak`, `opinion` (who to vote for), `pii` (personal data of named people beyond what the documents state), `harmful` |

2. Every case outside the Refusal and Safety groups that expects an answer SHALL have
   `evidence` checked verbatim (folded) against the text of at least one of its `docs` by a test
   that skips when `data/` is absent.

### Requirement 12: Corpus build and ingest suite

1. `scripts/build_eval_corpus.py` SHALL rebuild `data/parsed` from `eval/corpus_sources.txt`
   (visibility + URL or path under `data/sources/`) with the project's own preprocessing,
   pausing 1 s between web fetches, and print one line per source.
2. Scanned copies (`ocr` slice): image-only PDF and photo versions of real pages, made by a
   committed script from the sources, so the OCR path and its Arabic quality are measured.
3. An `ingest` suite SHALL drive `POST/GET/DELETE /api/v1/documents` end to end: new upload,
   identical re-upload (`duplicate`), same topic new bytes (replaced), visibility change, a
   scan, an empty file, a corrupt PDF, a renamed binary, an oversize file; then the deleted
   document is no longer retrieved and `corpus_version` changed at each step.
