# mcclub-rag

## Overview

RAG chatbot for the MC Club RAG Chatbot Challenge.

## Quickstart

```bash
cp .env.example .env
docker compose up -d --build
```

## Architecture

Layout (built: `ingest/`, `text/`, `retrieval/embed.py`, `retrieval/store.py`, the scripts;
the rest is planned):

```
src/mcclub_rag/
├── main.py, config.py, deps.py, schemas.py, errors.py, logging.py
├── answer.py, conversation.py, cache.py
├── api/        chat.py, documents.py, feedback.py, health.py
├── ingest/     parse.py, web.py, chunk.py, pipeline.py
├── text/       normalize.py, lang.py
├── retrieval/  embed.py, store.py, search.py, rerank.py, gate.py
└── llm/        client.py, quota.py, prompts.py
scripts/        download_models.py, ingest_folder.py, reindex.py
eval/           questions.jsonl, run_eval.py
tests/          conftest.py, test_api_contract.py, test_access.py, test_ingest.py, test_gate.py
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

## Evaluation

## Edge cases

See [EDGE_CASES.md](EDGE_CASES.md).
