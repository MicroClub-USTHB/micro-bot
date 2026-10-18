# Implementation Plan: Retrieval

Each task is test-first: write the listed tests, see them fail, then implement. Every pushed task
keeps CI green (`uv lock --check`, ruff, `pytest -m "not ocr and not tokenizer and not embedder
and not reranker"`, plus the models job). Unit tests use `AsyncQdrantClient(":memory:")`, the
helpers in `tests/pipeline_helpers.py` (fake dense encoder, real BM25 on tiny stopword files,
fake preprocessing) and a `FakeReranker` / fake ONNX session. Only tasks 4.2, 5.3 and 6.4 need
the real reranker. Commits stay under 200 changed lines; split a task across commits if needed.

- [x] 1. Settings, errors, registry and model download
- [x] 1.1 `RetrievalSettings` in `src/micro_bot/retrieval/settings.py`
  - Fields and defaults from design "Components / RetrievalSettings": `env_prefix="RETRIEVAL_"`;
    `top_k`, `top_n`, `gate_threshold`, `rerank_model`, `models_dir`, `qdrant_collection` read
    through `validation_alias` of `RETRIEVE_TOP_K`, `RERANK_TOP_N`, `GATE_THRESHOLD`,
    `RERANK_MODEL`, `MODELS_DIR`, `QDRANT_COLLECTION`; blank `GATE_THRESHOLD` -> None.
  - Validators: `prefetch_limit >= top_k`, `rerank_max_length >= 640`, `fusion in {rrf, dbsf}`.
  - `get_retrieval_settings()` with `lru_cache`.
  - Append each `RETRIEVAL_*` variable with a comment line to `.env.example`, and set
    `RERANK_MODEL=Alibaba-NLP/gte-multilingual-reranker-base` there.
  - `tests/test_retrieval_settings.py`: defaults; shared names picked up without prefix; blank
    threshold is None; `RETRIEVAL_PREFETCH_LIMIT=10` with `RETRIEVE_TOP_K=30` and
    `RETRIEVAL_RERANK_MAX_LENGTH=512` raise `ValidationError`.
  - _Requirements: 2.1, 4.2, 4.6, 5.4, 6.2, 7.1, 9.2_
- [x] 1.2 Errors and the shared Qdrant guard
  - `src/micro_bot/ingest/errors.py`: `RerankError` (`rerank_error`), `RerankerConfigError`
    (`reranker_config_error`); add both to `ALL_ERRORS`.
  - `src/micro_bot/retrieval/store.py`: rename `_guard` to public `guard` (update call sites).
  - Extend `tests/test_errors.py`: codes stable and unique.
  - _Requirements: 4.5, 8.1, 8.2_
- [x] 1.3 Reranker registry entries and download target
  - `retrieval/registry.py`: `RerankerSpec(repo, revision, onnx_file, onnx_sha256,
    tokenizer_sha256)` with `url()` and `slug` (directory name); `RERANKERS` keyed by both names:
    - gte: `onnx-community/gte-multilingual-reranker-base` @
      `ee64367e35a2db0da46bb6497e13a18f8bd585cb`, `onnx/model_int8.onnx`
      `ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35`, tokenizer
      `3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98`; keys
      `Alibaba-NLP/gte-multilingual-reranker-base`, `onnx-community/gte-multilingual-reranker-base`.
    - bge: `onnx-community/bge-reranker-v2-m3-ONNX` @
      `6f5ff65298512715a1e669753bc754d2bc8f367b`, `onnx/model_int8.onnx`
      `912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d`, tokenizer
      `8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3`; keys
      `BAAI/bge-reranker-v2-m3`, `onnx-community/bge-reranker-v2-m3-ONNX`.
  - `scripts/download_models.py`: `--only reranker` (+ `--model NAME`, default `RERANK_MODEL` or
    gte) into `dest/<slug>/` via `_fetch_verified`; included in "all" (so the Dockerfile bakes gte).
  - `tests/conftest.py`: `reranker_dir` session fixture (fails with the download command if
    missing) and auto-marker `reranker`; `pyproject.toml`: register the `reranker` marker.
  - `.github/workflows`: fast job excludes `reranker`; models job caches `models/reranker`
    (key on `registry.py`), downloads gte, runs `-m "ocr or tokenizer or embedder or reranker"`.
  - Extend the download-script tests: both keys resolve to the same spec; the reranker target
    writes `<slug>/tokenizer.json` and `<slug>/onnx/model_int8.onnx` (fake HTTP transport).
  - _Requirements: 4.1, 4.5, 11.2_

- [x] 2. Pure building blocks
- [x] 2.1 Follow-up query builder in `src/micro_bot/retrieval/query.py`
  - `build_search_query(message, history, max_words) -> tuple[str, bool]` per design.
  - `tests/test_retrieval_query.py`: no history; short follow-up joins the previous
    `user_message`; long message ignores history; only the last turn is used; non-`Turn` items
    ignored; whitespace/NFC normalised; empty message returns `("", False)`.
  - _Requirements: 1.5, 6.1, 6.2, 6.3, 6.4_
- [x] 2.2 Gate in `src/micro_bot/retrieval/gate.py`
  - `decide(scores, threshold) -> (top_score, decision)`.
  - `tests/test_gate.py`: empty -> `(0.0, "refuse")`; None -> answer; below / equal / above.
  - _Requirements: 1.4, 5.1, 5.2, 5.4, 5.5_
- [x] 2.3 Reranker passage text in `src/micro_bot/retrieval/rerank.py`
  - `passage(title, heading_path, text, *, header=True)`: header without consecutive repeats,
    `"\n\n"`, text; text alone when `header` is false.
  - Tests in `tests/test_rerank.py`: equals the chunk's `embed_text` for a chunk built by
    `chunk_document` with a short header (same header rule); repeats collapsed; header off.
  - _Requirements: 4.2_

- [x] 3. Hybrid search in `src/micro_bot/retrieval/search.py`
- [x] 3.1 `Candidate`, `visibility_filter`, `dedupe`
  - `tests/test_search.py`: public filter matches only `public`; member filter matches both;
    unknown role -> `ValueError`; `dedupe` keeps newest `ingested_at` per `(doc_id, index)`, tie
    goes to better fused rank, output keeps fused order.
  - _Requirements: 2.2, 3.1_
- [x] 3.2 `HybridSearch.search`
  - One `query_points` on the alias: dense + bm25 prefetch (`prefetch_limit`, filter on each),
    `RrfQuery(Rrf(k, weights))` or `FusionQuery(DBSF)`, `query_filter`, `limit=top_k`, payload
    fields only; payload -> `Candidate`; then `dedupe`. Qdrant errors through `store.guard`.
  - Tests on fake-ingested docs (`tests/pipeline_helpers.py`): a matching chunk comes back;
    **members-only chunks scoring higher on both arms never reach a public role** (and do reach
    a member); `top_k` respected; two versions of a doc during a replacement give one candidate
    per index; unequal weights are passed (spy client); a closed client -> `IndexUnavailable`.
  - _Requirements: 2.1, 2.2, 2.3, 2.5, 3.1, 3.2, 8.1, 11.3_
- [x] 3.3 `HybridSearch.neighbours`
  - One `scroll` for `index ± 1` of each hit's `(doc_id, content_hash)`, with the visibility
    filter.
  - Tests: neighbours of a middle chunk found; first/last chunk only one side; a neighbour of
    another version is not used; filter applied.
  - _Requirements: 7.1, 7.2_

- [x] 4. Reranker
- [x] 4.1 `OnnxReranker` and `get_reranker` in `retrieval/rerank.py`
  - sha256 checks via `embed._check_file` (group `reranker`) -> `RerankerConfigError`;
    `no_padding()`, `enable_truncation(max_length, strategy="only_second")`; one `session.run` per
    pair; `token_type_ids` only if declared; sigmoid; onnxruntime errors -> `RerankError`;
    `fingerprint` = repo@revision:sha256; warm-up pair at construction; `get_reranker` cached per
    (model, dir, threads, max_length); unknown model -> `RerankerConfigError` listing known names.
  - Tests with a fake session and a tiny tokenizer fixture: missing / wrong-hash files; one run
    per pair with batch size 1; inputs filtered to the declared ones; scores in (0, 1) and
    monotonic in the logit; error mapping; truncation keeps the whole query.
  - _Requirements: 1.3, 4.1, 4.2, 4.3, 4.4, 4.5, 8.2_
- [x] 4.2 Real-model tests (`reranker` marker) in `tests/test_rerank_model.py`
  - gte ranks a relevant passage above an unrelated one for an Arabic, a French and an English
    query; a pair scores identically alone and after scoring 5 other pairs; 30 x ~384-token
    pairs score in under 10 s.
  - _Requirements: 4.1, 4.3, 4.4, 9.1_
- [x] 4.3 Benchmark script `eval/bench_rerank.py`
  - Port the spec's probe: latency per thread count, batched vs batch-1 timings and score drift,
    RSS; reads the model from the registry and `MODELS_DIR`. No test (manual tool), but ruff-clean.
  - _Requirements: 9.2_

- [x] 5. Backend `src/micro_bot/retrieval/backend.py`
- [x] 5.1 `HybridRetriever.retrieve` core
  - Constructor with injected settings, encoders, client, reranker; `candidates()` (encode in
    `to_thread`, `HybridSearch.search`) and `rerank()` (lock + `to_thread`, `passage()` input);
    `retrieve()` = query -> candidates -> rerank -> top_n -> `gate.decide` -> `Retrieval`.
  - `tests/test_retrieval_backend.py` (FakeReranker = keyword overlap): chunks ordered by score,
    at most `top_n`; `chunk_id` = point id, title, text without header; `top_score`; threshold
    refuse vs answer with chunks still returned; empty query -> refuse with no Qdrant / reranker
    call; empty collection -> refuse; two concurrent `retrieve()` never overlap in `score()`.
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 2.4, 4.6, 5.1, 5.2, 5.3_
- [x] 5.2 Follow-ups, neighbours, logging, errors
  - Use `build_search_query` for search and rerank; neighbours when enabled (text joined in
    reading order, ids/scores/decision unchanged); `retrieval_done` structlog event; gate-open
    warning once at construction; encoder / reranker errors propagate typed.
  - Tests: a short follow-up finds the previous question's doc; neighbours on/off; log event has
    the fields and no query text (`structlog.testing.capture_logs`); gate-open warning;
    `EmbeddingError` / `RerankError` / `IndexUnavailable` propagate, no partial result.
  - _Requirements: 5.4, 6.1, 6.2, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3_
- [x] 5.3 `from_pipeline` and the end-to-end test
  - `HybridRetriever.from_pipeline(pipeline, settings=None, reranker=None)` reuses encoders,
    client and alias; default reranker from `get_reranker`.
  - Extend `tests/test_pipeline_e2e.py` (markers `embedder`, `reranker`): ingest `notes.md`,
    `retrieve()` returns its chunk first with `decision="answer"` for a member and nothing from a
    members-only doc for a public role.
  - _Requirements: 1.1, 2.3, 11.2_

- [x] 6. Evaluation harness
- [x] 6.1 Metrics in `eval/metrics.py` (+ `eval/__init__.py`)
  - `is_hit(chunk, question)` (folded evidence + doc), `hit_at(ranks, k)`, `mrr_at(ranks, k)`,
    `sweep(answerable_scores, unanswerable_scores)`, `suggest_threshold(sweep, max_false_refusal)`,
    `access_violations(rows)`.
  - `tests/test_eval_metrics.py`: hand-made rows for every function, folding (tatweel, case),
    the suggestion rule, no-hit MRR = 0.
  - _Requirements: 10.3, 10.4, 10.7_
- [x] 6.2 `eval/corpus.json` and `eval/questions.jsonl`
  - Corpus manifest for the 4 `data/parsed` files (visibility, topic). At least 40 questions per
    Requirement 10.2: >= 10 Arabic, >= 6 cross-lingual (both directions), >= 8 unanswerable,
    >= 4 follow-ups with `history`, >= 3 `access` (public role, members-only doc); evidence
    strings copied verbatim from the documents.
  - `tests/test_eval_questions.py`: schema of every line; unique ids; tag minimums; each `doc`
    is a topic in `corpus.json`; each evidence string occurs (folded) in that document's text;
    `access` questions point at members-only docs.
  - _Requirements: 10.1, 10.2_
- [x] 6.3 `eval/run_eval.py`
  - CLI per design "Evaluation harness": settings from env, in-memory Qdrant, real encoders +
    reranker, ingest corpus, per-question stages with timings, metrics pre/post rerank per tag,
    access violations, sweep + suggested threshold, latency median/p95, JSON to
    `eval/results/<UTC>-<label>.json`, table on stdout.
  - Test the orchestration with fakes (fake encoders, `FakeReranker`, 3 tiny docs, 4 questions):
    JSON written with settings, corpus version, rows, aggregates and sweep.
  - _Requirements: 9.1, 10.3, 10.4, 10.5, 10.6, 10.7_
- [x] 6.4 Measured defaults
  - Run the eval for: gte vs bge; `RETRIEVE_TOP_K` 10 / 15 / 20 / 30; RRF `k` 2 vs 60; DBSF;
    dense/sparse weights 1:1 vs 1:0.5; neighbours off/on. Commit the result JSONs.
  - Set the chosen `RETRIEVE_TOP_K`, fusion settings and `GATE_THRESHOLD` in `.env.example` and
    the `RetrievalSettings` defaults, with the measured numbers in their comments; make the
    model tests' latency bound match.
  - _Requirements: 5.1, 9.1, 10.6_

- [x] 7. Keep the repo docs in sync
  - `README.md` (retrieval section), `EDGE_CASES.md` (rows: batch-dependent int8 rerank scores;
    gte tokenizer padding; top-level filter propagation; RRF default k=2; app/ingest
    `IngestError` mismatch), `PROGRESS.md`, `AGENTS.md` ("Retrieval" built: what it is, rules that
    bite), my journal. Note for eequaled: change `config.py` `rerank_model` / `retrieve_top_k`
    defaults and map `index_unavailable` -> 503 in the wiring PR.
  - _Requirements: all (traceability)_
