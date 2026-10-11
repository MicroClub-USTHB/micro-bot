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
rate limiting and the daily LLM quota. Known limits, tracked in [PROGRESS.md](PROGRESS.md) and
[EDGE_CASES.md](EDGE_CASES.md): the `expires_at` upload field is accepted but ignored, the gate
and defaults are tuned on a small corpus (61 chunks), and answer quality with the live LLM has not
been evaluated yet (retrieval only).

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

- No framework: 1 LLM call per question, every call explicit.
- Qdrant hybrid search (dense + BM25 sparse), visibility filter inside the query.
- granite-311m-r2 as the default embedder; harrier-270m / bge-m3 / F2LLM-v2-330M to be A/B
  tested on the eval set (hit@5 and MRR, Arabic subset).
- Exact-match cache keyed by question + role + corpus version, instead of a semantic answer cache.
- Chunking ([spec](specs/chunking/)): structure first (headings), then paragraphs, lines and
  sentences (Arabic `؟ ؛ ۔` aware), size last. Target 300 / max 400 / min 80 **embedder tokens**,
  no overlap, tiny sections merged under their parent, and a "title > heading path" header
  prepended to the embedded and BM25 text. Semantic (embedding-based) chunking and overlap were
  rejected on published benchmarks; the sizes are settings so they can be A/B tested.

- Ingest pipeline ([spec](specs/ingest-pipeline/)): `ingest_file` / `ingest_url` →
  preprocess → chunk → granite int8 ONNX (CLS + L2, one sequence per run) + BM25 on folded
  text (no stemming, one ar+fr+en stopword set, IDF in Qdrant) → Qdrant. `doc_id = uuid5(topic)`;
  same content is `duplicate` (checked on the raw bytes before parsing), a new visibility is
  `updated`, new content `replaced` (new points first, old ones deleted after). Each version's
  index-0 point is written last, so listings and `corpus_version()` only see complete documents.
  Parsed documents are kept as JSON under `DATA_DIR/documents`, and a settings change rebuilds a
  new collection from them, then swaps the `QDRANT_COLLECTION` alias atomically.

- Retrieval ([spec](specs/retrieval/)): one Qdrant query with dense + BM25 prefetches fused by
  RRF (k=2), the visibility filter on every level, dedupe by document and chunk index, then the
  gte-multilingual-reranker-base int8 cross-encoder (one pair per run: batching moved scores by
  up to 1.8 logits) on the top 20, and a gate that refuses below a sigmoid score of 0.40 with
  **0 LLM calls**. Short follow-ups are searched together with the previous question (no LLM
  rewrite). bge-reranker-v2-m3 stays an A/B option: +0.014 MRR for 2.2x the latency.

## Evaluation

`uv run python -m eval.run_eval --label <name>` ingests `data/parsed` (the document part of each
`scripts/parse.py` dump, titled from `eval/corpus.json`) into an in-memory Qdrant
with the real models and runs the 48 questions of `eval/questions.jsonl` (Arabic, cross-lingual
both ways, unanswerable, follow-ups, access checks): hit@5 and MRR@10 before and after reranking,
per tag, a gate-threshold sweep and per-stage latency. Settings are environment overrides; results
and the comparison table are in [eval/results/](eval/results/README.md).

## Edge cases

See [EDGE_CASES.md](EDGE_CASES.md).
