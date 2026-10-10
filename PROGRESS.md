# progress

## 2026-10-10 (retrieval)

### Shipped

- Retrieval spec (`specs/retrieval/`) and all its tasks: `HybridRetriever` implements
  eequaled's `RetrievalBackend` (hybrid dense + BM25 in one Qdrant query, filter on every level,
  dedupe, gte int8 reranker one pair per run, gate, follow-up query, optional neighbours).
- Measured first: gte 3.3 s vs bge 7.7 s for 30 pairs; int8 rerank scores moved by up to 1.8
  logits in batches, so batch 1. `--only reranker` download, `reranker` test marker, CI step.
- Eval harness: 48 questions, `python -m eval.run_eval`, results in `eval/results/`. Defaults set
  from it: `RETRIEVE_TOP_K=20`, `RETRIEVAL_RRF_K=2`, `GATE_THRESHOLD=0.40` (gte). Median
  retrieval 1.3 s. 6 edge cases recorded.

### Blocked

- Wiring into `main.py` (eequaled): app error handler must also catch `ingest.errors.IngestError`
  and map `index_unavailable` to 503; `config.py` defaults for `rerank_model` / `retrieve_top_k`
  still say bge / 30; ownership of search/rerank/gate to confirm.

### Next

- `DocumentBackend` adapter and readiness probe, then the wiring PR with eequaled.
- Grow the corpus and question set: hit@5 is saturated at 1.0 on 70 chunks.

## 2026-10-10

### Shipped

- Ingest pipeline spec (`specs/ingest-pipeline/`) and all its tasks: `ingest_file` / `ingest_url`
  (async) → preprocess → chunk → dense + BM25 → Qdrant, plus `list_documents`,
  `delete_document`, `corpus_version`, startup checks and reindexing.
- Dense: granite int8 ONNX via onnxruntime, CLS + L2, one sequence per run (padded batches changed
  vectors by up to 0.012; batch 1 is also faster, 6.3 chunks/s on 4 threads).
- Sparse: fastembed BM25 on folded text, no stemming, one folded ar+fr+en stopword set, offline
  from pinned files; IDF computed by Qdrant; `avg_len` 110 measured on the corpus.
- Dedup on raw bytes before parsing; `updated` for a visibility change; replacement writes the new
  version before deleting the old; marker point written last; per-document locks; cleanup of
  half-written versions.
- Parsed documents kept as JSON in `DATA_DIR/documents`; a settings change rebuilds a new
  collection from them and swaps the alias atomically (`scripts/reindex.py` when startup reindex
  is off). `scripts/ingest_folder.py` for bulk loads.
- `download_models.py --only embedder|bm25`; new `embedder` test marker in the CI models job;
  an end-to-end test on the real models with a hybrid RRF query. 8 edge cases recorded.

### Blocked

- eequaled's approval of the pipeline seam (async, `title`/`topic` keywords, list/delete/version
  functions) and the bad-upload status question.

- Chunking spec (`specs/chunking/`), researched against published benchmarks (sentence-aware
  splitting best for Arabic RAG; no overlap; no semantic chunking; contextual headers).
- `chunk_document()`: heading paths rebuilt from the sections, oversized sections split at
  paragraphs → lines → sentences → words → tokens, tables/lists/code kept whole, small sections
  merged, "title > path" header for retrieval, exact token cap on the embedder tokenizer.
- Embedder `tokenizer.json` downloaded at build (pinned revision, sha256-verified); new `tokenizer`
  test marker, run in the CI models job.
- Real corpus check: VP candidacy PDF 40 sections → 10 chunks, Arabic article 11 → 23 chunks
  (132–340 tokens), all under the 400-token cap.

### Next

- Retrieval spec: hybrid query with RRF, reranker `max_length` ≥ 640, gate calibration.
- Eval: chunk target 200/300/400 × header on/off on `eval/questions.jsonl` (hit@5, MRR, Arabic subset).

## 2026-10-09

repo skeleton is up with project config and docker setup.
specced the eequaled api end to end. requirements plus design plus tasks. both sides build against one frozen boundary now.
ran the compatibility pass against the nisbenz preprocessing design. topic upsert and expiry land in the right spec.
wrote the jury contract and nisbenz boundary schemas first. both sides build against one file.
blocked on nothing. waiting on the nisbenz retrieval/store spec for the shared seam before wiring anything real.
next is config plus auth deps plus json error handlers. every failure speaks the jury shape from day one.

## 2026-10-10
- Repo skeleton, project config, Docker setup.
- Data preprocessing spec (`specs/data-preprocessing/`: requirements, design, tasks), all 12 tasks done.
- `preprocess_file()` / `preprocess_url()`: PDF, DOCX, XLSX, MD, TXT, images and web pages → one
  `ParsedDocument` (clean text, sections with page numbers, language, OCR flag, content hash).
- OCR with Tesseract (ara, fra, eng), only on pages without a text layer; startup fails if a language is missing.
- Web ingestion with httpx + trafilatura: boilerplate removed, size cap, SSRF check on every redirect.
- Text cleanup, header/footer removal, Arabic search folding, language detection,
  `detect_language()` for the answer layer.
- Typed ingest errors with stable codes for the API layer.
- Tessdata downloaded at image build, sha256-pinned.
- CI (GitHub Actions): ruff + pytest on every push and PR. 290 tests passing.
- Fixed: xberg dropped the last lines of Arabic OCR (forced page segmentation mode 3, regression-tested).
- 11 edge cases recorded in EDGE_CASES.md.

- Task 2: config, auth deps, and JSON error handlers committed (`8e2a357`). Every error matches `{"error":{"code","message"}}`.
- Task 3: structured logging middleware committed (`a36413b`). Emits JSON log per request with latency, token usage, and corpus version.
- Task 4: conversation memory committed (`dd178df`). Pins query language and keeps turn history with TTL.
- Task 5: exact-match cache committed (`f9da6ec`). Normalizes whitespace/greetings/Arabic, bypasses on conversation history.
- Task 6: single-call LLM client, daily UTC quota, and circuit breaker committed (`06e9eb8`). Hard 1-call ceiling, atomic reservation.
- Task 7: answer post-processing and extractive fallback committed (`e512e17`). Validates citations against retrieved chunks, fallback on quota/breaker exhaustion.
- Task 8: health, feedback, and documents endpoints committed (`533102b`). Admin auth, typed ingest errors, and degraded health probe handling.
- Task 9: rate limiter and chat orchestration committed (`16369e5`). Sliding-window per-IP limiter, language reuse, cache bypass, role privacy, and full jury contract verified.
- Tasks 1-9 marked as done in `localdocs/specs/eequaled-api/tasks.md`. All 73 tests green with zero OpenRouter network calls.

- Real Arabic scans needed in `tests/fixtures/` (`arabic_scan.pdf`, `arabic_photo.jpg`) to test real-world OCR quality.
- Waiting on eequaled: `ingest_file` / `ingest_url` as `async def`, and whether a bad upload returns
  a 4xx or 200 with `status: "failed"`.

### Next

- Push `nisbenz/data-preprocessing`, check CI on GitHub, open PR into `dev`.
- Spec the pipeline: chunking, embedding (granite ONNX), Qdrant hybrid store, `corpus_version()`.
