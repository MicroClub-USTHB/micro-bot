# Requirements Document

## Introduction

Retrieval turns a chat question into the few chunks the answer layer may use, and decides whether
the corpus can answer at all. It implements eequaled's frozen
`RetrievalBackend.retrieve(query, role, history) -> schemas.Retrieval` (`main.py`): hybrid dense +
BM25 search in Qdrant with the visibility filter inside the query, deduplication, a cross-encoder
reranker, and a gate that returns `decision="refuse"` when the best reranked chunk is too weak, so
the chat path answers "not grounded" with **0 LLM calls**. It also ships the evaluation harness
(`eval/questions.jsonl`, `eval/run_eval.py`) so fusion settings, the candidate count, the reranker
model and `GATE_THRESHOLD` are measured, not guessed.

Fixed decisions this spec builds on (AGENTS.md): plain Python, no RAG framework; at most 1 LLM
call per question, so **no LLM query rewriting**; Qdrant hybrid search (dense + BM25) through the
`QDRANT_COLLECTION` alias; the same encoder objects as ingest (`get_dense_encoder`,
`get_sparse_encoder`, no query prefix); reranker ONNX int8 on CPU via onnxruntime + tokenizers
(no PyTorch), sha256-pinned in `retrieval/registry.py`; offline at runtime.

**Measured before writing this (2026-10-10, 16-core dev machine, onnxruntime CPU, real corpus
passages, ar/fr/en queries; details in design.md):**

| | gte-multilingual-reranker-base int8 (341 MB) | bge-reranker-v2-m3 int8 (571 MB) |
|---|---|---|
| 30 pairs, ~384 tokens, batch 1, 4 threads | **3.3 s** (110 ms/pair) | 7.7 s (256 ms/pair) |
| 30 pairs, ~500 tokens, batch 1, 4 threads | 4.2 s | 10.1 s |
| same, 8 threads | 2.7 s (384 tok) | 6.8 s (384 tok) |
| batched (8 or 30) vs batch 1 | slower (5.1 to 5.5 s) | slower (10.9 to 11.2 s) |
| score change, batched vs alone | up to **1.8 logits** | up to **1.0 logit** |
| process RSS, batch 1 | 0.9 GB | 1.1 GB |

So: rerank **one pair per `session.run`** (deterministic, and faster), and the number of reranked
candidates is bounded by latency: 30 candidates is already 3.3 s with the faster model.

**In scope:** `retrieval/search.py` (hybrid query, filter, dedupe, neighbours), `retrieval/rerank.py`,
`retrieval/gate.py`, the `RetrievalBackend` implementation, a history-aware query builder, reranker
entries in `retrieval/registry.py` and `scripts/download_models.py`, `RETRIEVAL_*` settings,
`eval/questions.jsonl`, `eval/run_eval.py`, and tests. **Out of scope:** the `DocumentBackend`
adapter and readiness probe (separate task), wiring into `main.py` (eequaled's file, done together
later), prompts and answer formatting, and any change to `schemas.py`, `main.py`
or `config.py`.

## Requirements

### Requirement 1: Backend contract

**User Story:** As the chat endpoint (eequaled), I want `retrieve()` to return ranked, citable
chunks and a clear answer/refuse decision, so that I only spend an LLM call when the corpus can
answer.

#### Acceptance Criteria

1. The system SHALL provide a class implementing `async def retrieve(self, query: str, role: Role,
   history: list[object]) -> Retrieval` exactly as the `RetrievalBackend` protocol in `main.py`
   declares it, without changes to `main.py` or `schemas.py`.
2. WHEN chunks are returned THEN each `Chunk` SHALL have `chunk_id` = the Qdrant point id (string),
   `doc_id`, `title` (document title), `text` (the stored chunk text, no header) and `score` = the
   reranker score, ordered by `score` descending, at most `RERANK_TOP_N` of them.
3. `Chunk.score` and `top_score` SHALL be on a fixed 0..1 scale (sigmoid of the reranker logit),
   so `GATE_THRESHOLD` has the same meaning whatever the model's raw logit range.
4. `top_score` SHALL equal the highest returned `score`, or `0.0` when no chunk is returned.
5. IF the query is empty or whitespace only after normalisation THEN the system SHALL return no
   chunks, `top_score=0.0` and `decision="refuse"` without querying Qdrant or the reranker.
6. The system SHALL make no LLM call and no network call other than to Qdrant.

### Requirement 2: Hybrid search with access control inside the query

**User Story:** As the club, I want members-only documents never to reach a public user, even by
accident in a candidate list, and I want keyword and semantic matches combined.

#### Acceptance Criteria

1. WHEN `retrieve()` runs THEN the system SHALL send one Qdrant `query_points` request to the
   `QDRANT_COLLECTION` alias with two prefetches (dense vector `dense`, sparse vector `bm25`)
   fused server-side (RRF by default; the fusion method, RRF `k`, per-arm weights and per-arm
   prefetch limit SHALL be settings).
2. WHEN `role == "public"` THEN the filter `visibility == "public"` SHALL be attached to **each
   prefetch and to the fused query**. WHEN `role == "member"` THEN the filter SHALL allow
   `public` and `members`. An unknown role SHALL raise `ValueError` rather than default to member.
3. The dense query vector SHALL come from `get_dense_encoder(...).embed_query()` and the sparse one
   from `get_sparse_encoder(...).embed_query()` (the ingest objects, no query prefix).
4. Encoding and the reranker SHALL run off the event loop (`asyncio.to_thread`), so a chat request
   does not block other requests.
5. The fused result SHALL contain at most `RETRIEVE_TOP_K` candidates; these are the candidates
   handed to deduplication and the reranker.

### Requirement 3: Deduplication of candidates

**User Story:** As a user, I want five different passages, not the same passage twice because a
document was being replaced while I asked.

#### Acceptance Criteria

1. WHEN two candidates share `doc_id` and `index` (two versions during a replacement, or leftovers
   of a crashed ingest) THEN the system SHALL keep only the one with the newest `ingested_at`.
2. Deduplication SHALL happen before reranking, so duplicates cost no reranker time.

### Requirement 4: Reranking

**User Story:** As a user asking in Arabic, French or English, I want the passages that actually
answer my question at the top, not just the ones sharing words with it.

#### Acceptance Criteria

1. The system SHALL rerank candidates with a multilingual cross-encoder loaded from a sha256-pinned
   ONNX int8 file through onnxruntime + tokenizers. Both
   `onnx-community/gte-multilingual-reranker-base` and `onnx-community/bge-reranker-v2-m3-ONNX`
   SHALL be registry entries selectable by `RERANK_MODEL` alone (no code change). The default
   SHALL be gte (decided 2026-10-10: 2.3x faster, 250 MB less RAM); bge stays an A/B option.
2. The reranker input SHALL be the pair (query, `"title > heading path"` header + `"\n\n"` + chunk
   text), the same header the chunk was embedded with, truncated only on the passage side to
   `RETRIEVAL_RERANK_MAX_LENGTH` (default 640, at least 640) tokens.
3. The system SHALL score **one pair per `session.run`** (measured: batching changes scores by up
   to 1.8 logits and is slower), so a chunk's score does not depend on the other candidates.
4. The same (query, chunk) pair SHALL always get the same score (deterministic).
5. WHEN the model files are missing or have the wrong sha256 THEN startup SHALL fail with a typed
   config error naming the `download_models.py` command, as the embedder does.
6. At most one rerank SHALL run at a time per process (a lock), so concurrent chats do not
   oversubscribe the CPU; threads are a setting (default 4).

### Requirement 5: Gate

**User Story:** As the project, I want weakly supported questions refused without an LLM call, so
the 50-calls/day budget goes to questions the corpus can answer.

#### Acceptance Criteria

1. IF `GATE_THRESHOLD` is set AND `top_score < GATE_THRESHOLD` THEN `decision` SHALL be `"refuse"`.
2. IF no candidate survives search and deduplication THEN `decision` SHALL be `"refuse"`.
3. WHEN `decision == "refuse"` THEN the system SHALL still return the reranked chunks it has
   (the chat path ignores them; logs and the eval use them).
4. IF `GATE_THRESHOLD` is unset (not calibrated yet) THEN the gate SHALL answer whenever a
   candidate exists, and the backend SHALL log one warning at startup that the gate is open.
5. The gate SHALL be a pure function of the reranked scores and the threshold, testable without
   models.

### Requirement 6: Follow-up questions without an LLM rewrite

**User Story:** As a user, I want "and what about the deadline?" to find the same document as my
previous question, without the bot spending an LLM call to rewrite it.

#### Acceptance Criteria

1. WHEN `history` is empty THEN the search query SHALL be the message itself.
2. WHEN `history` is non-empty AND the message is short (fewer than
   `RETRIEVAL_FOLLOWUP_MAX_WORDS` words, default 8) THEN the search query and the rerank query
   SHALL be the previous turn's `user_message` followed by the current message.
3. WHEN the message is long enough to stand alone THEN the history SHALL be ignored.
4. The query builder SHALL be a pure function, tolerant of history items that are not `Turn`
   (they are ignored), because the protocol types `history` as `list[object]`.

### Requirement 7: Neighbour expansion (optional, off by default)

**User Story:** As the answer layer, I want a little surrounding context for a passage that starts
mid-argument, without retrieving bigger chunks.

#### Acceptance Criteria

1. WHEN `RETRIEVAL_NEIGHBOURS` is true THEN for each returned chunk the system SHALL fetch the
   chunks with `index ± 1` of the same `doc_id` and `content_hash` (one Qdrant request for all)
   and join their text around the hit in reading order in `Chunk.text`.
2. Expansion SHALL NOT change `chunk_id`, `score`, ordering or the gate decision, and SHALL NOT add
   a neighbour the role may not see (same visibility filter).
3. WHEN `RETRIEVAL_NEIGHBOURS` is false (default) THEN `Chunk.text` SHALL be the chunk alone.

### Requirement 8: Errors and observability

**User Story:** As an operator, I want a Qdrant or model failure to give a clear error and a log
line I can read, never document text in logs.

#### Acceptance Criteria

1. IF Qdrant is unreachable or errors THEN `retrieve()` SHALL raise `IndexUnavailable` (code
   `index_unavailable`, 503), the existing ingest error type.
2. IF the encoder or reranker fails at run time THEN `retrieve()` SHALL raise a typed error
   (`embedding_error` for encoders, a new `rerank_error` for the reranker) and SHALL NOT return a
   partial result.
3. The system SHALL log one structlog `retrieval_done` event per call with role, candidate count
   after dedupe, returned count, `top_score`, decision, whether the history was used, and
   per-stage milliseconds (encode, search, rerank). It SHALL NOT log the query or chunk text.

### Requirement 9: Performance

**User Story:** As a user, I want an answer in a few seconds on the club's CPU-only server.

#### Acceptance Criteria

1. WITH the default settings and ~300-token chunks THEN `retrieve()` SHALL take at most 3 s
   (median over the eval set, 4 threads, dev machine), measured by `run_eval.py`.
2. The reranker model, the candidate count (`RETRIEVE_TOP_K`) and the reranker threads SHALL be
   settings, so latency can be traded for recall without code changes.

### Requirement 10: Evaluation harness

**User Story:** As the team, I want one command that tells me whether a retrieval change helps,
in every language and on unanswerable questions, so defaults are measured.

#### Acceptance Criteria

1. `eval/questions.jsonl` SHALL hold at least 40 questions, each with an id, the question, its
   language, the role, `answerable`, tags, and for answerable ones the expected `doc` (topic) and
   one or more `evidence` strings found verbatim in the document. Labels SHALL NOT depend on chunk
   ids, so they survive rechunking.
2. The set SHALL include at least: 10 Arabic questions, 6 cross-lingual ones (question language
   differs from the document language, both directions), 8 unanswerable ones (on topic but not in
   the corpus), 4 follow-ups with a `history`, and 3 members-only documents asked as `public`
   (expected: not found).
3. `eval/run_eval.py` SHALL ingest a corpus directory into an in-memory Qdrant with the real models,
   run `retrieve()` for every question, and report hit@5 and MRR@10 (a hit = a chunk of the
   expected doc containing one of the evidence strings), overall and per tag, before and after
   reranking.
4. `run_eval.py` SHALL report the `top_score` distributions of answerable and unanswerable
   questions and sweep thresholds, printing for each the share of unanswerable questions refused
   and of answerable ones wrongly refused, and a suggested `GATE_THRESHOLD`.
5. `run_eval.py` SHALL report median and p95 `retrieve()` latency per stage.
6. Settings under test (reranker model, top-k, fusion, weights, chunking variants, neighbours)
   SHALL be passed as environment overrides, and each run SHALL write a JSON result with the
   settings, the corpus version and the metrics to `eval/results/`.
7. A public-role question SHALL count as a failure if any returned chunk is from a members-only
   document, whatever its rank.

### Requirement 11: Tests

1. Unit tests SHALL run without model files or Docker: in-memory Qdrant, the fake encoders from
   `tests/pipeline_helpers.py`, and a fake reranker session.
2. Tests needing the real reranker SHALL carry a new `reranker` pytest marker and run in the CI
   models job (which downloads the pinned reranker, sha256-checked).
3. There SHALL be a test that a public role never receives a members-only chunk, with members-only
   chunks scoring higher than public ones on both arms.

## Decisions and open questions

1. **Reranker default: gte** (decided 2026-10-10). bge stays a registry entry for the eval; switch
   only if it shows a clear hit@5/MRR gain worth ~4 s more per question. `config.py` still defaults
   `rerank_model` to bge, so the registry maps that name too and `.env.example` sets gte (ask
   eequaled to change the default).
2. **`RETRIEVE_TOP_K=30`** (eequaled's `config.py`) costs 3.3 s of reranking with gte. Accepted:
   `RETRIEVE_TOP_K` stays the rerank candidate count; the eval picks a new default (likely 15 to
   20), then eequaled is asked to change it.
3. **Ownership** of `search.py` / `rerank.py` / `gate.py` (unassigned in AGENTS.md): this spec
   assumes nisbenz builds them. Still to confirm with eequaled.
4. **`history` content:** accepted, `Turn`s only; cited `chunk_ids` would need a protocol change.
5. **CI cost:** accepted, the models job downloads gte (341 MB) once, cached like the embedder.
