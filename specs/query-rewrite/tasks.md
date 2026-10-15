# Implementation Plan: Query rewrite

Test-first per task; commits under 200 changed lines; CI stays green.

- [ ] 1. Rewrite call
  - [ ] 1.1 `llm/rewrite.py`: prompt, `parse_rewrite`, `QueryRewrite` (tests: fenced JSON, bad
    JSON, long text, unknown lang, duplicate translations)
  - [ ] 1.2 `QueryRewriter`: cache, timeout, failures as `None` (tests with a fake generator)
  - [ ] 1.3 Settings `REWRITE_ENABLED`, `REWRITE_TIMEOUT_S`, `REWRITE_CACHE_SIZE`;
    `DailyQuota.acquire(keep=...)`; `.env.example`
  - [ ] 1.4 Chat route: call, usage sum, protocol `rewrite=` kwarg, fakes; state wiring
- [ ] 2. Retrieval with a rewrite
  - [ ] 2.1 `HybridSearch.search` with extra dense arms (language filter) and `Candidate.language`
  - [ ] 2.2 `HybridRetriever.retrieve(..., rewrite=)`: arms, BM25 text, grouped rerank, log
- [ ] 3. Smaller models
  - [ ] 3.1 `EmbedderSpec` pooling / prefixes; registry entries; download script
  - [ ] 3.2 Download and check the four new models (embedder / reranker tests)
- [ ] 4. Eval
  - [ ] 4.1 `run_eval --rewrite`; e2e driver passes the rewrite on the retrieval layer
  - [ ] 4.2 `eval/bench_models.py` matrix + gate analysis
  - [ ] 4.3 E2E report: layers apart, language groups, attribution, failure traits
  - [ ] 4.4 Runs: record rewrites, matrix, e2e baseline vs rewrite; results README; PROGRESS,
    AGENTS.md, README
