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
3. The encoder SHALL batch inputs (sorted by length to limit padding), truncate at a configured
   `max_length` that is at least `INGEST_CHUNK_MAX_TOKENS + 2`, load the session once, and run in
   `asyncio.to_thread` so the event loop is not blocked.
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

