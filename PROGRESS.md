# Progress

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

### Shipped

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

### Blocked

- Real Arabic scans needed in `tests/fixtures/` (`arabic_scan.pdf`, `arabic_photo.jpg`) to test real-world OCR quality.
- Waiting on eequaled: `ingest_file` / `ingest_url` as `async def`, and whether a bad upload returns
  a 4xx or 200 with `status: "failed"`.

### Next

- Push `nisbenz/data-preprocessing`, check CI on GitHub, open PR into `dev`.
- Spec the pipeline: chunking, embedding (granite ONNX), Qdrant hybrid store, `corpus_version()`.
