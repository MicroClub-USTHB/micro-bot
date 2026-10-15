# Implementation Plan: Query rewrite

Test-first per task; commits under 200 changed lines; CI stays green.

- [x] 1. Rewrite call
  - [x] 1.1 `llm/rewrite.py`: prompt, `parse_rewrite`, `QueryRewrite` (tests: fenced JSON, bad
    JSON, long text, unknown lang, duplicate translations)
  - [x] 1.2 `QueryRewriter`: cache, timeout, failures as `None` (tests with a fake generator)
  - [x] 1.3 Settings `REWRITE_ENABLED`, `REWRITE_TIMEOUT_S`, `REWRITE_CACHE_SIZE`;
    `DailyQuota.acquire(keep=...)`; `.env.example`
  - [x] 1.4 Chat route: call, usage sum, protocol `rewrite=` kwarg, fakes; state wiring
- [x] 2. Retrieval with a rewrite
  - [x] 2.1 `HybridSearch.search` with extra dense arms (language filter) and `Candidate.language`
  - [x] 2.2 `HybridRetriever.retrieve(..., rewrite=)`: arms, BM25 text, grouped rerank, log
- [x] 3. Smaller models
  - [x] 3.1 `EmbedderSpec` pooling / prefixes; registry entries; download script
  - [x] 3.2 Download and check the four new models (embedder / reranker tests)
- [ ] 4. Eval
  - [x] 4.1 e2e `--rewrite` (chat and retrieval layers); the matrix makes and replays the
    rewrites itself, so `run_eval` keeps its typed-question role
  - [x] 4.2 `eval/bench_models.py` matrix + gate analysis
  - [x] 4.3 E2E report: layers apart, language groups, attribution, failure traits
  - [ ] 4.4 Runs: record rewrites, matrix, e2e baseline vs rewrite; results README; PROGRESS,
    AGENTS.md, README
