# progress

## 2026-10-15 (query rewrite, smaller models, eval report)

### Shipped (PR #21; eequaled's files changed with the user's go-ahead)

- Second LLM call per question, approved by the organizers: the query rewrite
  (`llm/rewrite.py`, `specs/query-rewrite/`). JSON with the language, a standalone corrected
  query and fr / en / ar translations; each translation searches its own language's chunks and
  the reranker scores each chunk against its language's text. Typed query kept; a failed rewrite
  falls back to the old path. Quota: the rewrite never takes the answer's last slot.
- Measured (eval/README.md): hit@5 0.909 -> 0.974, darija 0.66 -> 1.0, cross-lingual
  0.81 -> 0.98, typo variants losing the evidence 7/45 -> 1/45; e2e answers correct 82% -> 85%
  (darija 54% -> 83%), false refusals 11% -> 4%.
- Reranker default is now mmarco-mMiniLMv2-L12 (119 MB): hit@5 0.971 vs bge's 0.974 with the
  rewrite, rerank 2.7x faster; bge stays an A/B entry. Embedder stays bge-m3 (small ones lost 3-4
  points, and much more when the rewrite fails).
- `eval.bench_models` (embedder x reranker x typed/rewrite, gate analysis); e2e report with the
  layers apart (I29), quality per language group, failure causes and traits.
- `LLM_DISABLE_REASONING` for reasoning models (nemotron rewrite 20 s -> 2-3 s).
- Docs: `eval/README.md` is the eval guide (test types and sizes, labels, metrics, results,
  issue index). Run files and `eval/results/e2e/ISSUES.md` are no longer tracked.

### Blocked

- Rewrite latency on the real provider: `gemma-4-31b-it:free` was rate-limited upstream all day.
- p95 with a local rewrite was 9.3 s (> 8 s) with bge; mmarco should save ~1.1 s (not measured
  end to end).

### Next

- Snippet choice by the answer's words: 24 of the remaining failures are right answers whose
  cited snippet misses the evidence (EDGE_CASES.md).
- Run mmarco end to end; decide a gate for the rewrite path (t ~ 0.03 with mmarco).
- Review with eequaled: `api/chat.py`, `main.py`, `config.py`, `llm/client.py`, `llm/quota.py`.
- Check the labels of `dj-26`, `dj-27`.

## 2026-10-13 evening (leaks, link noise, prompt rules)

### Shipped

- Replies that repeat the system prompt or the passage block are withheld (I26, `08d38b3`):
  7 of 1,151 recorded replies flagged, all real leaks, no extra LLM call.
- Inlined `(url)` link targets dropped from embedded and reranked text, stored text unchanged
  (I30, `ddfaa8a`): 25% of corpus characters; Wikipedia facts reach rank 1; rerank p50 -10%,
  hit@5 0.847 -> 0.838. Reindex needed (`CHUNKER_VERSION` 2).
- Prompt rules (I27, I28, `8d04277`), measured on all 349 chat cases with only the prompt
  changed: passed 260 -> 265, false answers 9.4% -> 0%, every refuse slice passes; false
  refusals 6.6% -> 11.0% (8 of the 12 new ones had no evidence retrieved), +83 prompt tokens.
- Eval: corpus rebuilt (I31), labels E5/E6 fixed, answer text kept in results (I29).

### Blocked

- False refusals at 11% (≤ 10% target) are now mostly retrieval misses (19 of 30): cross-lingual
  and darija recall (I9).
- Production model still unmeasured (OpenRouter quota).

### Next

- Review the answer-layer changes (`answer.py`, `api/chat.py`, `llm/prompts.py`) with eequaled.
- Cross-lingual and darija recall; quality suite with the production model.

## 2026-10-13 (bge stack, NOT_FOUND, e2e re-run)

### Shipped

- Retrieval defaults: `BAAI/bge-m3` embedder and `BAAI/bge-reranker-v2-m3` on the top 10, gate
  off (darija hit@5 0.667 -> 0.771); chunks still sized in granite tokens.
- Refusals: the model starts a refusal with `NOT_FOUND` (not grounded, no sources); no more
  refusal-phrase matching (I25, `f38dd84`). On the 90 hard cases (`notfound-12b`, ollama gemma4
  12B): refuse-expected 16 -> 41 of 47 passed, 63 of 80 answered cases pass, language
  mismatches 35 -> 4, p95 6.8 s.
- `eval/results/e2e/ISSUES.md`: the re-run, failures by cause, new issues I30–I32 and E5.

### Blocked

- I26 (prompt and passage leak), I27 (example data as facts), I28 (partial refusals): the
  answer layer, eequaled's files.
- Generation with the production model not measured yet (OpenRouter key, 50 requests/day).

### Next

- I30: drop inline `(url)` link targets from the embedded / reranked text (Wikipedia chunks
  never reach the top 5), then reindex. I31: rebuild `data/parsed` with
  `scripts/build_eval_corpus.py` (predates the share-bar fix). E5: fix the `usthb-04` label.
- I29: report the chat and retrieval layers apart and keep the answer text in results.
- Quality suite with the production model (`--mode record`, small budget).

## 2026-10-12 (eval fixes, answer layer and the rest)

### Shipped

- On the user's request, eequaled's issues fixed on the same branch (to review with eequaled):
  the live app calls the LLM when a key is set (I1); the fallback no longer pastes chunks, it
  answers with a notice in the user's language and the passages as sources (I2–I4); refusals in
  the question's language (I5); citations `[doc_id: X]` / `[S1]` / `[X, Y]` accepted (I22);
  snippets around the best sentence (I6/I24); prompt answers across languages (I23); auth reads
  the app's settings (I20); `config.py` defaults aligned (I21); blank message skips retrieval
  (I18); `expires_at` kept, listed and enforced in every query (I14).
- Photo deskew before OCR (I13): 41% -> 99% word recall on the skewed photo; Pillow is now a
  runtime dependency.
- Defaults from measurements: `GATE_THRESHOLD=0.31` (I8), cross-language arm 20 (I9), query case
  and harakat normalization on (I7: flips 9.6% -> 6.6%).
- System e2e suites: contract 22/22, faults 9/9, cache 5/5, ingest 16/16.

### Blocked

- I17 / I15 (upload title): the frozen API. I9 target (0.85) and typos (I7) not reached.

### Next

- Full e2e with an LLM (`--suite all`, record mode, the production model) to measure I23 and the
  answer-level numbers; reindex deployments (payload schema 2).

## 2026-10-12 (eval fixes, retrieval and ingest side)

### Shipped

- `specs/eval-fixes/` (requirements, design, tasks) covering all 24 issues of
  `eval/results/e2e/ISSUES.md`, with an owner per issue.
- Retrieval: a short message is merged with history only on a follow-up cue (I10); query
  normalization settings (I7); optional cross-language dense arm (I9: cross-lingual hit@5
  0.755 -> 0.776, default off, target 0.85 not met); reranker session pool and rerank cut (I19);
  optional penalty for instruction-like chunks; gate documented as a relevance floor (I8).
- Ingest: share-bar pages are `empty_document` (I16); PDF title from page-1 headings when the
  metadata title is generic (I15, MRR 0.750 -> 0.770); `instruction_like` payload flag, payload
  schema 2 (reindex needed).
- `text/snippet.best_snippet` for eequaled (I6/I24: evidence shown 72% -> 82% offline).
- Eval: gate-score distributions, `--watch`, per-stage load times, `eval.calibrate_gate`
  (offline sweep), `eval.bench_load`. Results in `eval/results/README.md`.

### Blocked

- I13 OCR warning: no usable signal from xberg; needs a new image dependency (user's call).
- Answer-layer issues (I1–I6, I17, I18, I20–I23) and `expires_at` (I14): eequaled's files.

### Next

- Rerun the e2e suite (`--suite all --label fixes`) to confirm I10, the noise settings, the
  injection cases and the load suite; lower the gate with eequaled's I1/I2 fixes.

## 2026-10-11 (e2e eval built and run)

### Shipped

- `eval/e2e/` runner: rule-based scoring (contract, behaviour, gold, evidence, must_not,
  access, language, path), in-process and HTTP drivers, recording / replay / budgeted
  generator, perturbations, suites (quality, robustness, contract, faults, cache, ingest, load,
  restart), optional cached LLM judge, scorecard (`thresholds.json`), compare.
- Corpus grown to 63 documents (55 real incl. 5 scans, 8 adversarial), 269 cases over 42
  slices; labels checked against the documents.
- First full run (`eval/results/e2e/20261011T203423Z-full-1`): 0 access leaks, contract 22/22,
  faults 9/9, cache 5/5, ingest 15/16; chat 113/269. All 21 issues with evidence in
  `eval/results/e2e/ISSUES.md`.

### Blocked

- Generation quality: the live app never calls the LLM (I1, eequaled) and no key locally.
- HTTP target and restart suite: Docker not running on the dev machine.

### Next

- With eequaled: I1 (wire the generator), I2/I3 (fallback must not paste unrelated or injected
  text), I5 (refusal language), I14 (`expires_at`).
- nisbenz: I7/I8 gate recalibration with noisy variants, I9 cross-lingual distractors, I10 short
  topic switch, I13 OCR deskew, I16 share-bar pages.

## 2026-10-11 (e2e eval spec)

### Shipped

- `specs/e2e-eval/` (requirements, design, tasks): an end-to-end harness over the chat API with
  rule-based scoring, canary/injection corpus, record/replay of LLM calls, robustness, faults,
  cache, load and restart suites, and a threshold scorecard.

### Blocked

- The running app never calls the LLM: `main.default_state()` keeps `FakeGenerationService`, and
  `OpenRouterClient.generate` is sync vs the async protocol (EDGE_CASES.md). eequaled's fix;
  generation quality cannot be measured live until then.

### Next

- Build the harness task by task (`specs/e2e-eval/tasks.md`).

## 2026-10-11 (0.1.0 review)

### Shipped

- AGENTS.md and README.md brought up to date with the wired app (PR #11): the adapter, probe and
  lifespan are in `main.py`, the pipeline error mapping is fixed (EDGE_CASES updated).
- Release check on `dev` (0199b4d): lockfile, ruff, 735 unit tests and the 83 model tests pass;
  CI green. Smoke test on the real models with a real Qdrant (uvicorn + Qdrant container):
  upload, duplicate, unsupported type, listing, in-corpus question answered with sources,
  members-only document hidden from public and shown to a member, off-topic question refused.

### Blocked

- Not verified: `docker compose up --build` (port 8000 was taken on the dev machine), a live
  OpenRouter call (no key), real Arabic scan fixtures, answer quality (generation) on the eval set.

### Next

- Resolve `expires_at`, align `config.py` defaults with the retrieval defaults (gte / 20 / 0.40).
- Run the compose build and a live LLM chat once, then tag `v0.1.0` on `main`.
- Grow the corpus and question set; re-measure `INGEST_BM25_AVG_LEN`.

## 2026-10-11 (chunking eval)

### Shipped

- Found that the eval corpus (`data/parsed`) is `scripts/parse.py` dumps and was ingested whole
  (metadata dict and section list indexed, every title "Metadata"). The eval now ingests only the
  document text with titles from `eval/corpus.json`: 61 chunks instead of 70.
- Chunk size x header comparison (Req 9 of the chunking spec): 300 / 400 with the header kept,
  reranked MRR 0.986; header off costs 0.025 to 0.039 MRR at every size.

### Next

- Re-measure `INGEST_BM25_AVG_LEN` on the clean chunks (110 was measured on the dumps).
- Re-ingest the local dev Qdrant from document text with a persistent `DATA_DIR`.
- Neighbour / section expansion for the LLM context, judged on answers (not MRR).

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
