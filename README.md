# mcclub-rag

## Overview

RAG chatbot for the MC Club RAG Chatbot Challenge. Version **0.1.0**: upload documents (files or
web pages, Arabic / French / English, scans included), ask questions in any of the three languages,
get a grounded answer with sources, with member-only documents hidden from public callers. Plain
Python (FastAPI, Qdrant, ONNX models on CPU), no RAG or agent framework, offline except the
explicit URL fetch and the one OpenRouter call per answered question.

## Quickstart

```bash
cp .env.example .env   # set OPENROUTER_API_KEY, MEMBER_TOKEN, ADMIN_API_KEY
docker compose up -d --build   # downloads and sha256-checks the models at build time
```

The API listens on port 8000. Upload a document, then ask a question:

```bash
curl -H "X-Admin-Key: $ADMIN_API_KEY" -F file=@bylaws.pdf -F visibility=public \
  localhost:8000/api/v1/documents
curl -H 'content-type: application/json' -d '{"message":"Quelle est la cotisation ?"}' \
  localhost:8000/api/v1/chat
```

Development: `uv sync --locked`, `uv run pytest -m "not ocr and not tokenizer and not embedder and
not reranker"` (no model files needed), `uv run ruff check . && uv run ruff format --check .`.

## Status (0.1.0)

Working end to end on the real models and a real Qdrant: ingestion (PDF, images with OCR, text,
web), hybrid retrieval with reranking, the refusal gate, role-based visibility, exact-match cache,
rate limiting, the daily LLM quota and a query rewrite (second LLM call). Known limits, tracked in
[PROGRESS.md](PROGRESS.md), [EDGE_CASES.md](EDGE_CASES.md) and [eval/README.md](eval/README.md):
answers were evaluated with a local gemma4 12B, not yet with the production model (349 cases,
2026-10-15: retrieval hit@5 0.97, 85% of answerable questions pass every rule check, 4% false
refusals, 3% false answers); cited snippets show the evidence 87% of the time (target 90%: the
snippet is picked by the question's words); arabizi questions are answered in French or English.

## Architecture

Layout:

```
src/mcclub_rag/
├── main.py, config.py, deps.py, schemas.py, errors.py, logging.py, rate_limit.py
├── answer.py, conversation.py, cache.py
├── api/        chat.py, documents.py, feedback.py, health.py
├── ingest/     preprocess.py, parse.py, ocr.py, web.py, chunk.py, pipeline.py, docstore.py, ...
├── text/       normalize.py, lang.py, sentences.py
├── retrieval/  registry.py, embed.py, store.py, settings.py, query.py, search.py, rerank.py,
│               gate.py, backend.py
└── llm/        client.py, quota.py, prompts.py
scripts/        download_models.py, ingest_folder.py, reindex.py
eval/           questions.jsonl, corpus.json, metrics.py, run_eval.py, bench_rerank.py, results/
tests/          one test module per source module, plus contract, access and wiring tests
```

### API contract

- Endpoints: `POST /api/v1/chat`, `GET /api/v1/health`, `POST/GET /api/v1/documents`,
  `DELETE /api/v1/documents/{doc_id}`, `POST /api/v1/feedback`.
- Auth: members send `Authorization: Bearer <MEMBER_TOKEN>`; admins send
  `X-Admin-Key: <ADMIN_API_KEY>`.
- Chat response: `answer`, `conversation_id`, `grounded`, `sources[{doc_id, title, snippet}]`,
  `usage{llm_calls, prompt_tokens, completion_tokens}`, `latency_ms` (integer).
  A `message` longer than 2000 chars returns 422.
- Errors are always JSON: `{"error": {"code": "...", "message": "..."}}`.

## Key decisions

- No framework: at most 2 LLM calls per question (query rewrite + answer), every call explicit.
- Qdrant hybrid search (dense + BM25 sparse), visibility filter inside the query.
- bge-m3 embedder + mmarco-mMiniLMv2-L12 reranker (119 MB) on the top 10 candidates, with the
  query rewrite: hit@5 0.974 with bge-reranker-v2-m3, 0.971 with mmarco at 2.7x less rerank
  time; small embedders lost 3-4 points ([results](eval/results/README.md), 2026-10-15).
- Exact-match cache keyed by question + role + corpus version, instead of a semantic answer cache.
- Chunking ([spec](specs/chunking/)): structure first (headings), then paragraphs, lines and
  sentences (Arabic `؟ ؛ ۔` aware), size last. Target 300 / max 400 / min 80 **embedder tokens**,
  no overlap, tiny sections merged under their parent, and a "title > heading path" header
  prepended to the embedded and BM25 text. Semantic (embedding-based) chunking and overlap were
  rejected on published benchmarks; the sizes are settings so they can be A/B tested.

- Ingest pipeline ([spec](specs/ingest-pipeline/)): `ingest_file` / `ingest_url` →
  preprocess → chunk → bge-m3 int8 ONNX (CLS + L2, one sequence per run) + BM25 on folded
  text (no stemming, one ar+fr+en stopword set, IDF in Qdrant) → Qdrant. `doc_id = uuid5(topic)`;
  same content is `duplicate` (checked on the raw bytes before parsing), a new visibility is
  `updated`, new content `replaced` (new points first, old ones deleted after). Each version's
  index-0 point is written last, so listings and `corpus_version()` only see complete documents.
  Parsed documents are kept as JSON under `DATA_DIR/documents`, and a settings change rebuilds a
  new collection from them, then swaps the `QDRANT_COLLECTION` alias atomically.

- Retrieval ([spec](specs/retrieval/)): one Qdrant query with dense + BM25 prefetches fused by
  RRF (k=2), the visibility filter on every level, dedupe by document and chunk index, then the
  mmarco-mMiniLMv2-L12 int8 cross-encoder (one pair per run: batching moved bge scores by up to
  1.8 logits) on the top 10, each chunk scored against the rewrite in its own language. The relevance-floor gate is off: bge scores right darija matches near
  0, so no threshold works, and the LLM refuses unknowns (with gte it refused below 0.31). Short follow-ups with a
  follow-up cue ("and…", "it", "وكم") are searched together with the previous question (no LLM
  rewrite). Without the LLM (no key, budget spent, provider error) the answer is a notice in the
  user's language with the best passages as sources, never pasted chunks. gte stays an A/B option.

## Evaluation

`uv run python -m eval.run_eval --label <name>` ingests `data/parsed` (the document part of each
`scripts/parse.py` dump, titled from `eval/corpus.json`) into an in-memory Qdrant
with the real models and runs the 128 questions of `eval/questions.jsonl` (Arabic, darija, arabizi, cross-lingual
both ways, unanswerable, follow-ups, access checks): hit@5 and MRR@10 before and after reranking,
per tag, a gate-threshold sweep and per-stage latency. Settings are environment overrides; results
and the comparison table are in [eval/results/](eval/results/README.md).

### End-to-end eval

`uv run python -m eval.e2e.run --suite all --label <name>` drives the real chat and documents API
(in-process, or `--target http` against a running app) with 349 cases over 68 documents: 42
edge-case slices (cross-lingual, Darija, Arabizi, OCR, tables, near-duplicates, false premises,
follow-ups, access, prompt injection in questions, documents and history, system-prompt leaks,
PII, harmful requests), plus perturbation, API contract, fault, cache, ingest and load suites.
Scoring is rule-based (canary and marker strings for leaks), with an optional LLM judge. See
[specs/e2e-eval/](specs/e2e-eval/) and the eval guide [eval/README.md](eval/README.md) (test
types and sizes, labels, metrics, latest results, issue index).

## Edge cases

See [EDGE_CASES.md](EDGE_CASES.md).
