# Requirements Document

## Introduction

The ingest pipeline is the last ingest stage. It takes a file or a URL, runs preprocessing
(`specs/data-preprocessing/`) and chunking (`specs/chunking/`), embeds every chunk (dense + BM25
sparse), and stores the result in Qdrant so that retrieval can run hybrid search with the
visibility filter inside the query. It also owns document identity (`doc_id` from the topic),
deduplication and replacement, deletion and listing for the admin endpoints, and the
`corpus_version()` that keys the answer cache.

Fixed decisions this spec builds on (AGENTS.md): granite-embedding-311m-multilingual-r2 shipped
ONNX int8 via onnxruntime + tokenizers (no PyTorch), CLS pooling + L2 normalize, no query prefix,
768 dims; Qdrant in its own container with a persistent volume; BM25 sparse via fastembed; offline
at runtime (`HF_HUB_OFFLINE=1`); no LLM calls at ingest.

**In scope:** `ingest/pipeline.py`, `retrieval/embed.py` (dense + sparse encoders, used by retrieval
for queries too), `retrieval/store.py` (collection, upsert, delete, list), the parsed-document store
under `DATA_DIR`, `corpus_version()`, reindexing, `scripts/ingest_folder.py`, the ONNX model target
in `scripts/download_models.py`, and the tests. **Out of scope:** hybrid query, fusion, reranking and
the gate (retrieval spec), HTTP endpoints and status codes (eequaled's API side), and changes to
preprocessing or chunking.

## Requirements

### Requirement 1: Entry points and result contract

**User Story:** As the API (eequaled), I want two stable calls that ingest a file or a URL and tell
me what happened, so that the documents endpoint is a thin wrapper.

#### Acceptance Criteria

1. The system SHALL expose `async def ingest_file(path, filename, visibility, *, title=None,
   topic=None) -> IngestResult` and `async def ingest_url(url, visibility, *, title=None,
   topic=None) -> IngestResult`. The keyword-only arguments are additions to the frozen seam and
   need eequaled's approval (see open questions).
2. WHEN ingestion succeeds THEN `IngestResult` SHALL hold `doc_id`, `status` (`created`,
   `replaced`, `duplicate` or `updated`), `title`, `source`, `source_type`, `visibility`,
   `language`, `chunk_count`, `content_hash`, `ocr_used`, `warnings` and `duration_ms`.
3. IF any stage fails THEN the system SHALL raise a typed `IngestError` subclass with a stable
   `code` (the existing preprocessing codes plus the new ones in Requirement 9) and SHALL NOT
   return a partial result. Mapping codes to HTTP responses stays on the API side.
4. IF chunking returns no chunks THEN the system SHALL raise `EmptyDocument`.
5. WHEN `ingest_file` is called THEN the system SHALL check the file size before reading it and
   raise `FileTooLarge` without loading an oversized file into memory.

### Requirement 2: Document identity

**User Story:** As an admin, I want re-uploading a newer version of the same document to replace
the old one, so that the bot never answers from two versions of the same rules.

#### Acceptance Criteria

1. The system SHALL derive `doc_id = uuid5(NAMESPACE, topic)` with a fixed, committed namespace
   UUID, so the same topic gives the same `doc_id` on every machine and run.
2. WHEN no `topic` is given THEN the topic SHALL be the filename stem for files, and for URLs the
   final URL after redirects, lower-cased scheme and host, without query string, fragment or
   trailing slash.
3. WHEN a `topic` is given THEN the system SHALL strip it, apply NFC, and reject an empty or
   longer-than-200-character topic with `InvalidInput`.
4. Each chunk's Qdrant point id SHALL be `uuid5(doc_id, f"{content_hash}:{index}")`, so that two
   versions of a document never overwrite each other's points.

### Requirement 3: Deduplication, replacement and visibility changes

**User Story:** As a club member, I want answers to stay available while a document is being
re-ingested, and I never want to see a half-written document.

#### Acceptance Criteria

1. IF the store already holds a complete copy of `doc_id` with the same `content_hash` and the same
   visibility THEN the system SHALL return `status="duplicate"` without embedding anything.
2. IF the copy has the same `content_hash` but another visibility THEN the system SHALL update the
   visibility of all its points in place and return `status="updated"` without re-embedding.
3. IF `doc_id` exists with another `content_hash` THEN the system SHALL write all new points first
   and only then delete the old version's points, and return `status="replaced"`. There SHALL be no
   moment where the document has zero points.
4. A copy SHALL count as complete only when the number of stored points with that `doc_id` and
   `content_hash` equals the `chunk_count` stored in their payload. IF an earlier ingest crashed
   halfway THEN the next ingest of the same content SHALL rewrite it, not report `duplicate`.
5. WHEN two ingests of the same `doc_id` run at the same time THEN the system SHALL serialize them
   (per-`doc_id` lock), so that the result equals running them one after the other.

### Requirement 4: Dense embedding

**User Story:** As the retrieval stage, I want document and query vectors from exactly the same
encoder, so that cosine scores are meaningful across Arabic, French and English.

#### Acceptance Criteria

1. The system SHALL embed each chunk's `embed_text` with granite-embedding-311m-multilingual-r2
   (`onnx/model_quint8_avx2.onnx` at the pinned revision) through onnxruntime and the same
   `tokenizer.json` the chunker uses, take the CLS vector and L2-normalize it to 768 floats.
2. The encoder SHALL expose `embed_documents(texts)` and `embed_query(text)`; queries get no
   prefix, so both SHALL produce identical vectors for identical text.
3. The encoder SHALL run one sequence per inference call (no padded batches), truncate at a
   configured `max_length` that is at least `INGEST_CHUNK_MAX_TOKENS + 2`, load the session once,
   and run in `asyncio.to_thread` so the event loop is not blocked. Measured on the int8 model:
   padded batching changed vectors by up to 0.012 per component (cosine 0.994 to the same text
   alone) because activations are quantized per call, and it was slower (5.2 vs 6.3 chunks/s).
4. IF the ONNX file is missing or its sha256 is wrong at startup THEN the system SHALL fail fast
   with an instruction naming the path and the download command.
5. `scripts/download_models.py --only embedder` SHALL download the ONNX file and the tokenizer at
   the pinned revision and verify both sha256s, like the tessdata target.
6. WHEN the same text is embedded twice, alone or inside a batch THEN the vectors SHALL agree
   within 1e-4 per component.

### Requirement 5: BM25 sparse vectors

**User Story:** As a member asking with exact words (a name, an acronym, an article number), I want
keyword matching that works on Arabic with or without diacritics and on accented French.

#### Acceptance Criteria

1. The system SHALL compute BM25 sparse vectors with fastembed on
   `fold_for_search(chunk.embed_text)`, and the same encoder SHALL serve queries
   (`fold_for_search(query)`).
2. The sparse encoder SHALL be language-agnostic: no stemming, and one stopword set that is the
   union of the Arabic, French and English lists, so that a French query still matches a mixed or
   Arabic chunk's shared terms (names, numbers, acronyms).
3. The sparse side SHALL run fully offline: any fastembed files it needs SHALL be baked into the
   image by `download_models.py`, and a missing file SHALL fail fast at startup.
4. The Qdrant sparse vector SHALL use the IDF modifier, so that IDF is computed by Qdrant over the
   whole collection and stays correct as documents are added and removed.

### Requirement 6: Qdrant collection and payload

**User Story:** As the retrieval stage, I want one collection with a known schema, so that hybrid
search and the visibility filter can be written once.

#### Acceptance Criteria

1. The collection SHALL have a named dense vector `dense` (768, cosine) and a named sparse vector
   `bm25`. `ensure_collection()` SHALL create it if missing and SHALL be idempotent.
2. Each point's payload SHALL hold `doc_id`, `content_hash`, `index`, `chunk_count`, `text`,
   `heading_path`, `page_start`, `page_end`, `section_index`, `language`, `token_count`,
   `visibility`, `title`, `source`, `source_type` and `ingested_at`. It SHALL NOT hold `embed_text`.
3. The system SHALL create payload indexes on `doc_id`, `content_hash`, `visibility` and `index`.
4. Every write SHALL wait for Qdrant's acknowledgement (`wait=True`) before the next step, so a
   returned `IngestResult` means the document is searchable.

### Requirement 7: Document management and the parsed-document store

**User Story:** As an admin, I want to list and delete documents, and I want a change of embedder
or chunking settings not to require re-uploading or re-OCRing everything.

#### Acceptance Criteria

1. The system SHALL expose `async list_documents() -> list[DocumentInfo]` (one entry per complete
   document: `doc_id`, `title`, `source`, `source_type`, `visibility`, `language`, `chunk_count`,
   `content_hash`, `ingested_at`) and `async delete_document(doc_id) -> bool` (False if unknown).
   These are new seam functions for eequaled's endpoints.
2. WHEN a document is ingested THEN the system SHALL save its `ParsedDocument` as JSON under
   `DATA_DIR/documents/<doc_id>.json` (written atomically) after the points are stored, and
   `delete_document` SHALL remove both the points and that file.
3. The stored JSON SHALL be enough to rebuild the document's points without the original file or
   URL (no re-parse, no re-OCR, no re-fetch).

### Requirement 8: Corpus version and reindexing

**User Story:** As the answer cache, I want a version string that changes whenever the indexed
content or how it was indexed changes, so that I never serve an answer built from stale chunks.

#### Acceptance Criteria

1. The system SHALL provide `index_signature()`: a hash of the embedder id, revision and ONNX
   sha256, the sparse encoder settings, the collection schema version and `chunking_signature()`.
2. The system SHALL provide `corpus_version()`: a hash of `index_signature()` and the sorted
   `(doc_id, content_hash, visibility)` of every complete document. It SHALL change after every
   create, replace, visibility update and delete, and SHALL stay the same otherwise, including
   across restarts and across processes (the CLI and the API).
3. The physical collection SHALL be named after `index_signature()` and reached through the
   `QDRANT_COLLECTION` alias. WHEN the signature changes THEN `reindex()` SHALL rebuild a new
   collection from the parsed-document store and switch the alias atomically, so that search keeps
   working on the old collection until the new one is complete.
4. WHEN the app starts and the alias points to a collection with another signature THEN the system
   SHALL reindex before accepting ingests, IF `INGEST_REINDEX_ON_STARTUP` is true (default), and
   otherwise fail fast with an instruction to run the reindex script.

### Requirement 9: Errors, concurrency, observability and performance

**User Story:** As an operator, I want failures to be typed and logged without document content,
and ingestion not to starve the chat endpoint on a small CPU.

#### Acceptance Criteria

1. IF Qdrant is unreachable or rejects a write THEN the system SHALL raise `IndexUnavailable`
   (code `index_unavailable`, suggested 503). IF the embedder session fails at inference THEN it
   SHALL raise `EmbeddingError` (code `embedding_error`). Startup model problems SHALL raise
   `EmbedderConfigError` (code `embedder_config_error`).
2. IF a failure happens after some new points were written THEN the system SHALL delete those
   points (best effort) and leave the previous version untouched.
3. Whole-document ingestion SHALL be limited by `INGEST_MAX_CONCURRENCY`, and CPU-bound stages
   (chunking, dense, sparse) SHALL run in `asyncio.to_thread`. onnxruntime intra-op threads SHALL
   be configurable (`INGEST_EMBED_THREADS`).
4. WHEN a document is ingested THEN the system SHALL log one `document_ingested` event (doc_id,
   status, chunk count, per-stage durations in ms, source without query string) or one
   `document_ingest_failed` event (code, stage). It SHALL NOT log document or chunk text.
5. WHEN 32 chunks of about 300 tokens are encoded on 4 CPU threads THEN dense + sparse encoding
   SHALL finish in under 10 s (measured 6.3 chunks/s, so about 65 s for a 400-chunk document; a
   performance test with the real model, marked like the tokenizer tests).

### Requirement 10: Bulk ingestion

**User Story:** As the team, I want to load the initial corpus from a folder in one command.

#### Acceptance Criteria

1. `scripts/ingest_folder.py <dir> --visibility public|members` SHALL ingest every supported file
   in the folder through `ingest_file`, print one line per file (status or error code), and exit
   non-zero if any file failed.
2. WHEN it is run twice on the same folder THEN the second run SHALL report every file as
   `duplicate` and change nothing.

## Open questions

- **eequaled:** approve `async def` for the seam, the keyword-only `title`/`topic` additions, and the
  new `list_documents`/`delete_document` functions. Bad uploads: the pipeline raises typed errors;
  4xx vs 200 `status:"failed"` stays an API decision.
- Should `DATA_DIR` also keep the raw uploaded bytes (to allow re-parsing after a preprocessing
  fix), or is the parsed JSON enough? The draft keeps only the JSON.
- Startup reindex can take minutes on CPU for a large corpus. Is blocking startup acceptable, or
  should the app serve from the old collection and reindex in the background?
