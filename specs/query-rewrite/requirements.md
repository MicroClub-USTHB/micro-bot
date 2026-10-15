# Requirements: Query rewrite (second LLM call), smaller models, eval report

Decided with the user on 2026-10-15: the organizers allow a second LLM call per question. It
rewrites every question before retrieval, so typos, darija, arabizi, follow-ups and
cross-language questions reach the encoders as clean text. With clean queries, smaller
multilingual embedders and rerankers may be enough; the eval decides. The user gave the
go-ahead to change eequaled's files (`llm/`, `api/chat.py`, `main.py`, `config.py`) for this.

## Req 1: Rewrite call

1.1 Every chat question that reaches retrieval gets one rewrite call first (when enabled), on
the same generation service and model as the answer.
1.2 The call returns JSON: `lang` (`ar`, `darija`, `arabizi`, `fr`, `en`), `query` (the
question made standalone with the previous user turn, typos fixed, in its own language) and
translations `fr`, `en`, `ar` (Modern Standard Arabic).
1.3 Names, acronyms, numbers, dates, codes and URLs are copied exactly; the call never answers
the question or adds facts. The user text is data, not instructions.
1.4 An invalid reply (not JSON, missing `query`, any text over 500 characters) is ignored:
retrieval runs as before the rewrite existed. So does a provider failure, a timeout
(`REWRITE_TIMEOUT_S`), an open circuit breaker, or a quota with fewer than 2 free slots (the
answer call keeps its slot).
1.5 Rewrites are cached in memory by (message, previous user message); a cache hit costs no call.
1.6 `REWRITE_ENABLED=false` turns the call off (the app then behaves as before this spec).
1.7 `usage` in the chat response counts both calls (calls and tokens); a rewrite-cache hit
costs no call.

## Req 2: Retrieval with a rewrite

2.1 The original query keeps its arms (dense, BM25, follow-up merge, normalization), so a bad
rewrite can add candidates but never remove the ones the original finds.
2.2 Each distinct rewrite text adds a dense arm: `query` over all chunks; each translation over
the chunks of its language (`RETRIEVAL_REWRITE_LANG_FILTER`, default on). The BM25 arm searches
the original and the rewrite texts together. The cross-language arm is not used with a rewrite.
2.3 The reranker scores each candidate against the translation in the chunk's language, or
`query` when there is none (`und` chunks). Still one pair per run, at most `RERANK_TOP_K` pairs.
2.4 The visibility filter is on every arm (as before).
2.5 `retrieval_done` logs whether a rewrite was used and the number of arms (never the text).

## Req 3: Smaller models

3.1 New registry entries, sha256-pinned, ONNX, permissive licenses: embedders
`granite-embedding-97m-multilingual-r2` (98 MB, CLS) and `multilingual-e5-small` (118 MB, mean
pooling, `query: ` / `passage: ` prefixes); rerankers `mmarco-mMiniLMv2-L12-H384-v1` (119 MB,
multilingual) and `ms-marco-MiniLM-L6-v2` (23 MB, English only, to test the "English is enough"
idea).
3.2 The encoder supports mean pooling and query/passage prefixes from the spec; the prefix is
part of the index signature.
3.3 Defaults change only if the matrix (Req 4.2) shows the smaller pair with the rewrite at
most 2 points of hit@5 under the current pair with the rewrite on every language group, and a
lower p95.

## Req 4: Eval

4.1 Retrieval eval with rewrites: `eval.run_eval --rewrite record|replay` (local ollama or
OpenRouter), rewrites recorded once and replayed for every model.
4.2 Model matrix (`eval.bench_models`): every embedder x reranker, rewrite off and on, one ingest
per embedder; hit@5, MRR, per language group; encode / rerank / ingest timings; gate analysis.
4.3 Gate analysis: for each combination, how well the top reranker score separates "evidence
retrieved" from "no evidence / unanswerable" (ROC AUC, threshold sweep), so we know whether a
gate is worth turning on.
4.4 E2E report: a scorecard (retrieval hit, answer correctness, refusals, citations, safety,
LLM calls, p50 / p95), the same per language group (fr, en, ar, darija, arabizi; plus
cross-lingual), failure attribution (retrieval miss, generation miss, citation miss), and the
common traits of failed cases. Chat and retrieval layers are never mixed in one rate.
4.5 The `llm_calls <= 1` check becomes `<= 2`.
