# Implementation Plan: Ingest Pipeline

Each task is test-first: write the listed tests, see them fail, then implement. Every pushed task
keeps CI green (`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer and not embedder"`,
plus the models job). Unit tests use `FakeDenseEncoder` (8-dim vectors from a hash of the text),
the real `Bm25SparseEncoder` on tiny stopword files in `tmp_path`, `AsyncQdrantClient(":memory:")`
and a `tmp_path` docstore. Only task 4.3 needs the real ONNX model.

- [x] 1. Settings, errors, result models and the model registry
- [x] 1.1 Pipeline settings in `src/mcclub_rag/ingest/settings.py`
  - Add the fields from design "Data Models": `qdrant_url`, `qdrant_collection`, `embed_model`,
    `embed_dim`, `models_dir`, `data_dir` (read through `validation_alias` of the shared names), and
    `qdrant_timeout_s=10`, `embed_threads=4`, `embed_max_length=512`, `bm25_k=1.2`, `bm25_b=0.75`,
    `bm25_avg_len=110`, `upsert_batch=64`, `reindex_on_startup=True`.
  - Extend the validator: `embed_max_length >= chunk_max_tokens + 2`.
  - Append each new `INGEST_*` variable, with one comment line, to `.env.example`.
  - Extend `tests/test_settings.py`: defaults; `QDRANT_URL` and `DATA_DIR` are picked up without the
    prefix; `INGEST_EMBED_MAX_LENGTH=300` with the default chunk cap raises `ValidationError`.
  - _Requirements: 4.3, 5.1, 8.4, 9.3_
- [x] 1.2 New errors in `src/mcclub_rag/ingest/errors.py`
  - `IndexUnavailable` (`index_unavailable`), `EmbeddingError` (`embedding_error`),
    `EmbedderConfigError` (`embedder_config_error`), `IndexStale` (`index_stale`); add them to
    `ALL_ERRORS`.
  - Extend `tests/test_errors.py`: codes are stable and unique.
  - _Requirements: 1.3, 9.1_
- [x] 1.3 Result models in `src/mcclub_rag/ingest/models.py`
  - `IngestStatus`, `IngestResult`, `DocumentInfo`, `StoredDocument` as in design "Data Models".
  - Extend `tests/test_models.py`: frozen; `StoredDocument` JSON round trip keeps a
    `ParsedDocument` with sections and `created_at` intact.
  - _Requirements: 1.2, 7.1, 7.3_
- [x] 1.4 `src/mcclub_rag/retrieval/registry.py` and download targets
  - `EmbedderSpec` and `EMBEDDERS` with the granite entry (revision `44399559…`,
    `onnx/model_quint8_avx2.onnx`, sha256 `f1fdd44e7e1ac51f12ab7957c7bd092e064d596c288513bf9d326842f669edee`,
    tokenizer sha256 `0087c868…`, dim 768, `cls`); `BM25_REPO = "Qdrant/bm25"`,
    `BM25_REVISION = "22b8d2af71a76161e18dd432d2cee0eefa66e412"`, `BM25_FILES`:
    - `arabic.txt` `0df01c0a184d8c15077c6f0ee70e25e0c0308d827b92b14a19f5f819a0c465d0`
    - `french.txt` `58ac7f7c074b70dc0fc86f6fcf40b2c227e9f95627e90e57971d5cce1e30e2e9`
    - `english.txt` `019f104ba2ed07436d05f9cdd3383034ad66014edc27fc651f837e1a038b6451`
  - `scripts/download_models.py`: import these constants (drop the duplicated ones); add
    `download_embedder(dest)` (tokenizer + ONNX into `dest/onnx/`, reusing `_fetch_verified`) and
    `download_bm25(dest)`; `--only embedder` and `--only bm25`; keep `--only tokenizer` working.
  - Extend `tests/test_download_models.py` (mocked transport): both new targets verify sha256,
    leave no file on mismatch, skip valid existing files; CLI routing.
  - _Requirements: 4.1, 4.5, 5.3_

- [x] 2. Document identity in `src/mcclub_rag/ingest/identity.py`
  - Commit a fixed `NAMESPACE` UUID; `file_topic`, `url_topic`, `clean_topic`, `doc_id_for`,
    `point_id` as in the design.
  - `tests/test_identity.py`: golden `doc_id` for two topics; `report.v2.pdf` → `report.v2`;
    URL cases (upper-case host, `:443`, query, fragment, trailing slash, root path); NFC; empty and
    201-character topics raise `InvalidInput`; `point_id` differs across hashes and indexes and is
    stable.
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Parsed-document store in `src/mcclub_rag/ingest/docstore.py`
  - `DocStore(root)` with `save` (tmp + fsync + `os.replace`), `load`, `delete`, `iter_all`,
    `set_visibility`; async wrappers via `asyncio.to_thread`.
  - `tests/test_docstore.py`: round trip; overwrite; delete of a missing id is fine; `*.tmp` files
    are ignored by `iter_all`; a corrupt file is reported, not raised from `iter_all`;
    `set_visibility` rewrites only that field.
  - _Requirements: 7.2, 7.3_

- [x] 4. Encoders in `src/mcclub_rag/retrieval/embed.py`
- [x] 4.1 `SparseEncoder` protocol and `Bm25SparseEncoder`
  - Check files against `expected`, fold the stopword union, build fastembed `Bm25` with
    `specific_model_path=cache_dir=model_dir`, `disable_stemmer=True`; `embed_documents` and
    `embed_query` fold first and return `SparseVector`s; `SPARSE_VERSION = 1`; `fingerprint`.
  - `tests/test_embed.py`: `الإنتخابات` in a document matches the query `الانتخابات`; `Règlement`
    matches `reglement`; a folded stopword (`إلى`) is dropped; missing file or wrong hash →
    `EmbedderConfigError`; fingerprint changes with `avg_len`; no network (`HF_HUB_OFFLINE=1`).
  - _Requirements: 5.1, 5.2, 5.3_
- [x] 4.2 `DenseEncoder` protocol and `OnnxDenseEncoder`
  - sha256 checks of ONNX and tokenizer, `EMBED_DIM` check, session options, own truncating
    tokenizer, one sequence per `run`, CLS + L2 normalize, `EmbeddingError` wrapping;
    `get_dense_encoder` / `get_sparse_encoder` cached per settings.
  - `tests/test_embed.py`: missing ONNX file, wrong hash (small fake file with its hash passed in),
    unknown `EMBED_MODEL`, and `EMBED_DIM` mismatch all raise `EmbedderConfigError` with the
    download command in the message.
  - _Requirements: 4.1, 4.2, 4.3, 4.4_
- [x] 4.3 Real-model tests (`embedder` marker)
  - Register `embedder` in `pyproject.toml`; `onnx_path` fixture in `tests/conftest.py` that fails
    with the download instruction when the file is missing and auto-applies the marker.
  - `tests/test_embed_model.py`: shape 768 and unit norm; a text alone vs inside a list of
    different-length texts agree within 1e-4; `embed_query(x) == embed_documents([x])[0]`; the
    FR↔EN translation pair scores above the unrelated sentence; perf: 32 chunks of about 300
    tokens, dense + sparse, under 20 s with 4 threads (GitHub runner: 11 s).
  - _Requirements: 4.1, 4.2, 4.3, 4.6, 9.5_
- [x] 4.4 CI in `.github/workflows/ci.yml`
  - Fast job: `-m "not ocr and not tokenizer and not embedder"`. Models job: cache `models/embedder`
    and `models/bm25` keyed on `src/mcclub_rag/retrieval/registry.py`, download
    `--only embedder` and `--only bm25`, set `MODELS_DIR`, run `-m "ocr or tokenizer or embedder"`.
  - _Requirements: 4.5, 5.3_

- [x] 5. Qdrant store in `src/mcclub_rag/retrieval/store.py`
- [x] 5.1 Collection, schema and alias
  - `VectorStore.ensure`, `create_physical`, `switch_alias`, payload indexes, error wrapping into
    `IndexUnavailable`.
  - `tests/test_store.py` (in-memory client): `ensure` returns `created` then `ok`, and `stale`
    after a signature change; a real collection named like the alias raises; `switch_alias` drops
    the old collection and a search through the alias still works; a client raising a transport
    error becomes `IndexUnavailable`.
  - _Requirements: 6.1, 6.3, 8.3, 9.1_
- [x] 5.2 Writes and marker reads
  - `upsert` (batched, `wait=True`), `versions`, `count`, `set_visibility`, `delete_version`,
    `delete_other_versions`, `delete_doc`, `markers` (newest `ingested_at` per `doc_id`).
  - `tests/test_store.py`: two versions of a doc → `markers` returns the newest only;
    `delete_other_versions` keeps exactly one hash; `set_visibility` touches every point of the doc;
    a visibility-filtered query sees only matching points; payload has no `embed_text`.
  - _Requirements: 3.2, 6.2, 6.4, 7.1_

- [x] 6. Pipeline in `src/mcclub_rag/ingest/pipeline.py`
- [x] 6.1 `IngestPipeline`, `init_pipeline`, `close_pipeline`, `index_signature`
  - Construct from settings with injectable encoders, client and docstore; lazy module singleton;
    `SCHEMA_VERSION = 1`. Test helpers in `tests/pipeline_helpers.py` (`FakeDenseEncoder`, a
    pipeline factory on the in-memory client).
  - `tests/test_pipeline.py`: `init_pipeline` creates the collection and alias; signature is stable
    and changes with a chunking setting, the dense fingerprint, or the sparse fingerprint.
  - _Requirements: 8.1, 8.3_
- [x] 6.2 `ingest_file` happy path and `_index`
  - Size check before reading, read in a thread, topic → `doc_id`, preprocess, chunk, encode, build
    points, upsert `1..n-1` then `0`, save `StoredDocument`, return `created`.
  - Tests: a Markdown file → `created`, `chunk_count` points with the payload fields, JSON saved;
    oversize file raises `FileTooLarge` without `read_bytes` being called (spy); a document with no
    chunks → `EmptyDocument` and nothing stored; bad visibility → `InvalidInput` before I/O.
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.1, 6.2, 6.4, 7.2_
- [x] 6.3 Duplicate, visibility update and replacement
  - `_check_existing` before preprocessing for files; `delete_other_versions` after the save.
  - Tests: same file twice → `duplicate` and preprocessing not called the second time (spy); same
    file with another visibility → `updated`, store and JSON both changed, no encoder calls; new
    content same topic → `replaced`, only new points remain; a store spy shows the point count of
    the doc never reaches 0 during the replacement; points of a crashed half-ingest (no marker, or
    count below `chunk_count`) are rewritten, not reported `duplicate`.
  - _Requirements: 3.1, 3.2, 3.3, 3.4_
- [x] 6.4 Concurrency, failure cleanup and logging
  - Semaphore, per-`doc_id` locks, reindex write lock (shared by ingests), cleanup on failure after
    the first upsert, `document_ingested` / `document_ingest_failed` events with stage timings.
  - Tests: two concurrent ingests of the same topic with different content end with exactly one
    version and both results consistent with a serial order; an encoder that raises mid-way leaves
    the previous version intact and no new points; a store raising on upsert → `IndexUnavailable`
    and old version intact; `capture_logs` shows one event with the keys and no document text.
  - _Requirements: 3.5, 9.1, 9.2, 9.3, 9.4_
- [x] 6.5 `ingest_url`
  - Preprocess first (mocked fetch, existing helper), topic from the final URL, then the shared path.
  - Tests: query string and fragment do not change `doc_id`; a redirect uses the final URL as topic;
    same page twice → `duplicate`; logged source has no query string.
  - _Requirements: 1.1, 2.2, 3.1_
- [x] 6.6 `list_documents` and `delete_document`
  - Tests: lists one `DocumentInfo` per complete doc (not the half-ingested one); delete removes
    points and JSON and returns True, then False; deleted doc disappears from the list.
  - _Requirements: 7.1, 7.2_
- [x] 6.7 `corpus_version`
  - Tests in `tests/test_corpus_version.py`: changes after create, replace, visibility update and
    delete; unchanged after a duplicate; equal for two pipeline instances on the same client;
    changes with `index_signature`.
  - _Requirements: 8.2_
- [x] 6.8 `reindex` and startup handling, `scripts/reindex.py`
  - Build a new physical collection from `DocStore`, keep `doc_id` / visibility / `ingested_at`,
    switch the alias only if every document succeeded; `init_pipeline` reindexes on `stale` or
    raises `IndexStale` when `reindex_on_startup=False`; thin CLI calling `reindex()`.
  - `tests/test_reindex.py`: after a chunking-setting change, startup rebuilds without calling
    preprocessing (spy), search works through the alias, the old collection is gone, the corpus
    version changes; one corrupt JSON → alias unchanged and `reindex()` raises; reindex off →
    `IndexStale`; an ingest waits while a reindex holds the write lock.
  - _Requirements: 7.3, 8.3, 8.4_

- [x] 7. Bulk ingestion in `scripts/ingest_folder.py`
  - Walk the folder (sorted, non-recursive unless `--recursive`), `ingest_file` each file, print
    `status` or the error code per file, exit 1 if any failed.
  - `tests/test_ingest_folder.py` (pipeline factory injected): first run `created` for each file,
    second run all `duplicate`; one unsupported file → its error code printed and exit code 1.
  - _Requirements: 10.1, 10.2_

- [x] 8. Repo docs kept in sync
  - `EDGE_CASES.md` rows with evidence from the design probes: int8 batch dependence; BM25 offline
    resolution in fastembed; stopwords compared after folding; cross-language stopword collisions;
    `avg_len` 256 vs 110; crash between upsert and JSON save; marker-less leftover points.
  - `README.md`: ingest pipeline subsection (flow, marker point, alias reindex, corpus version).
  - `AGENTS.md`: seam list (`async`, keyword-only `title`/`topic`, `list_documents`,
    `delete_document`, `corpus_version`, `init_pipeline`), the `embedder` marker and commands, and
    the "Not yet built" list.
  - `PROGRESS.md`: dated entry.
  - _Requirements: all (keeps repo docs in sync per AGENTS.md)_
