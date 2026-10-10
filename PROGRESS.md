# progress

## 2026-10-09

repo skeleton is up with project config and docker setup.
specced the eequaled api end to end. requirements plus design plus tasks. both sides build against one frozen boundary now.
ran the compatibility pass against the nisbenz preprocessing design. topic upsert and expiry land in the right spec.
wrote the jury contract and nisbenz boundary schemas first. both sides build against one file.
blocked on nothing. waiting on the nisbenz retrieval/store spec for the shared seam before wiring anything real.
next is config plus auth deps plus json error handlers. every failure speaks the jury shape from day one.

## 2026-10-10

- Task 2: config, auth deps, and JSON error handlers committed (`8e2a357`). Every error matches `{"error":{"code","message"}}`.
- Task 3: structured logging middleware committed (`a36413b`). Emits JSON log per request with latency, token usage, and corpus version.
- Task 4: conversation memory committed (`dd178df`). Pins query language and keeps turn history with TTL.
- Task 5: exact-match cache committed (`f9da6ec`). Normalizes whitespace/greetings/Arabic, bypasses on conversation history.
- Task 6: single-call LLM client, daily UTC quota, and circuit breaker committed (`06e9eb8`). Hard 1-call ceiling, atomic reservation.
- Task 7: answer post-processing and extractive fallback committed (`e512e17`). Validates citations against retrieved chunks, fallback on quota/breaker exhaustion.
- Task 8: health, feedback, and documents endpoints committed (`533102b`). Admin auth, typed ingest errors, and degraded health probe handling.
- Task 9: rate limiter and chat orchestration committed (`16369e5`). Sliding-window per-IP limiter, language reuse, cache bypass, role privacy, and full jury contract verified.
- Tasks 1-9 marked as done in `localdocs/specs/eequaled-api/tasks.md`. All 73 tests green with zero OpenRouter network calls.

