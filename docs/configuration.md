# Configuration reference

Every environment variable the app reads, with its default and why. All of them have the
shown default in code (`config.py`, `ingest/settings.py`, `retrieval/settings.py`), so `.env` only
needs the few lines in [.env.example](../.env.example); add a line here only to override.

```bash
# OpenRouter API key (required for LLM calls)
OPENROUTER_API_KEY=
# OpenRouter model id used for answer generation
OPENROUTER_MODEL=google/gemma-4-31b-it:free
# Bearer token that grants members-level access to /api/v1/chat
MEMBER_TOKEN=
# Key sent as X-Admin-Key for document management endpoints
ADMIN_API_KEY=
# Qdrant endpoint (service name inside docker compose)
QDRANT_URL=http://qdrant:6333
# Qdrant collection holding the chunks
QDRANT_COLLECTION=mcclub
# Dense embedding model (Hugging Face id)
EMBED_MODEL=BAAI/bge-m3
# Dense vector size; must match EMBED_MODEL
EMBED_DIM=1024
# Cross-encoder reranker model (key in retrieval/registry.py). mmarco (119 MB): with the query
# rewrite as good as BAAI/bge-reranker-v2-m3 (571 MB) at 2.7x less time; bge is better when the
# rewrite fails
RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# Where model weights and tessdata live inside the container
MODELS_DIR=/app/models
# Persistent app data directory (uploads, documents, quota state)
DATA_DIR=/data
# Max LLM requests per day (OpenRouter free tier budget)
LLM_DAILY_LIMIT=50
# Requests held back from the daily budget as a safety margin
LLM_RESERVE=5
# LLM request timeout in seconds
LLM_TIMEOUT_S=25
# true: ask OpenRouter to turn reasoning off (a reasoning model took ~20 s per rewrite, 2-3 s
# without); needed when OPENROUTER_MODEL is a reasoning model
LLM_DISABLE_REASONING=false
# Rewrite each question with one extra LLM call before retrieval (typos, darija, arabizi,
# follow-ups, translations); false = retrieve with the question as typed
REWRITE_ENABLED=true
# Rewrite call timeout in seconds; on timeout the question is searched as typed
REWRITE_TIMEOUT_S=6
# Rewrites kept in memory (by question + previous question)
REWRITE_CACHE_SIZE=512
# Candidates reranked after hybrid search (eval: 20 keeps MRR of 30, rerank 1.3 s vs 1.9 s)
RETRIEVE_TOP_K=20
# Chunks kept after reranking and passed to the LLM
RERANK_TOP_N=5
# Minimum rerank score (0..1, sigmoid) to answer; empty = open gate (the LLM refuses unknowns).
# Empty with bge: it scores right darija matches near 0, so no threshold works (eval/results,
# 2026-10-13). With gte, 0.31 (python -m eval.calibrate_gate, 2026-10-12).
GATE_THRESHOLD=
# Per-client-IP /api/v1/chat requests allowed in one sliding window
CHAT_RATE_LIMIT_REQUESTS=20
# Per-IP chat rate-limit sliding-window duration in seconds
CHAT_RATE_LIMIT_WINDOW_S=60
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# --- Data preprocessing (parsing, OCR, web) ---
# Directory holding ara/fra/eng .traineddata (set in the Dockerfile; override for local dev)
# TESSDATA_PREFIX=/app/models/tessdata
# OCR languages, JSON list; each needs <lang>.traineddata in TESSDATA_PREFIX
INGEST_OCR_LANGUAGES=["ara","fra","eng"]
# PDF pages with fewer non-whitespace chars than this are OCR'd
INGEST_OCR_MIN_CHARS_PER_PAGE=32
# Max upload size in MB
INGEST_MAX_FILE_MB=25
# Max pages per document
INGEST_MAX_PAGES=300
# Per-document hang guard in seconds
INGEST_PARSE_TIMEOUT_S=60
# Extra seconds of budget per PDF page or image (OCR time)
INGEST_OCR_PAGE_TIMEOUT_S=10
# Straighten a skewed photo (1-5°) before OCR; a 2.2° phone photo went from 41% to 99% word recall
INGEST_OCR_DESKEW=true
# Documents preprocessed at the same time
INGEST_MAX_CONCURRENCY=1
# Web fetch timeout in seconds
INGEST_WEB_TIMEOUT_S=15
# Max web response size in bytes
INGEST_WEB_MAX_BYTES=5242880
# Max redirects followed per URL
INGEST_WEB_MAX_REDIRECTS=5
# User-Agent sent when fetching URLs
INGEST_WEB_USER_AGENT=mcclub-rag/0.1 (+document ingestion)
# Languages py3langid may choose from, JSON list
INGEST_LANG_CODES=["ar","fr","en"]
# Below this many letters a text is tagged "und"
INGEST_LANG_MIN_CHARS=20
# Below this probability a text is tagged "und"
INGEST_LANG_MIN_PROB=0.5
# A second language above this share of characters makes a document "mixed"
INGEST_LANG_MIXED_SHARE=0.2
# Reply language when a chat message has no letters (ar, fr or en)
INGEST_LANG_DEFAULT=fr

# --- Chunking (sizes in embedder tokens) ---
# Chunk size the packer aims for
INGEST_CHUNK_TARGET_TOKENS=300
# Hard cap per chunk, header included (reranker max_length must be >= ~1.3x this + query)
INGEST_CHUNK_MAX_TOKENS=400
# Sections smaller than this are merged with their siblings
INGEST_CHUNK_MIN_TOKENS=80
# Max tokens of the "title > heading path" header (must be < max / 4)
INGEST_CHUNK_MAX_HEADER_TOKENS=48
# Prepend the title and heading path to the embedded / BM25 text (true/false)
INGEST_CHUNK_CONTEXT_HEADER=true
# Embedder tokenizer.json used to count tokens (default: $MODELS_DIR/embedder/tokenizer.json)
# INGEST_CHUNK_TOKENIZER_PATH=/app/models/embedder/tokenizer.json

# --- Ingest pipeline (embedding, Qdrant store; also reads QDRANT_*, EMBED_*, MODELS_DIR, DATA_DIR) ---
# Qdrant request timeout in seconds
INGEST_QDRANT_TIMEOUT_S=10
# onnxruntime intra-op threads for the embedder (more than 4 barely helps)
INGEST_EMBED_THREADS=4
# Embedder truncation length in tokens (must be >= INGEST_CHUNK_MAX_TOKENS + 2)
INGEST_EMBED_MAX_LENGTH=512
# BM25 term-frequency saturation k1
INGEST_BM25_K=1.2
# BM25 length normalization b (0 = off, 1 = full)
INGEST_BM25_B=0.75
# Average chunk length in BM25 tokens (measured mean on the corpus)
INGEST_BM25_AVG_LEN=110
# Points per Qdrant upsert request
INGEST_UPSERT_BATCH=64
# Rebuild the index at startup when embedder/chunking settings changed (false = fail fast)
INGEST_REINDEX_ON_STARTUP=true
# Server-side fusion of the dense and BM25 arms: rrf or dbsf
RETRIEVAL_FUSION=rrf
# RRF ranking constant k (eval: 2 beat 60 before reranking, MRR 0.861 vs 0.816)
RETRIEVAL_RRF_K=2
# RRF weight of the dense arm
RETRIEVAL_DENSE_WEIGHT=1.0
# RRF weight of the BM25 arm
RETRIEVAL_SPARSE_WEIGHT=1.0
# Candidates fetched per arm before fusion (>= RETRIEVE_TOP_K)
RETRIEVAL_PREFETCH_LIMIT=40
# Extra dense candidates in languages other than the question's (cross-lingual recall); 0 = off
RETRIEVAL_CROSS_LANG_PREFETCH=20
# Reranker truncation length in tokens (>= 640; only the passage is truncated)
RETRIEVAL_RERANK_MAX_LENGTH=640
# onnxruntime threads for the reranker
RETRIEVAL_RERANK_THREADS=4
# Reranker sessions run in parallel for concurrent users (threads are split between them; ~350 MB RAM each)
RETRIEVAL_RERANK_WORKERS=1
# Rerank only the first k fused candidates (empty = all RETRIEVE_TOP_K); bge on 10 = same hits, half the time
RETRIEVAL_RERANK_TOP_K=10
# A follow-up shorter than this many words is searched together with the previous question
RETRIEVAL_FOLLOWUP_MAX_WORDS=8
# ...but only when it has a follow-up cue (and/et/و, it/il/ذلك, a bare "Where?"); false merges every short message
RETRIEVAL_FOLLOWUP_CUES=true
# Lower-case an ALL-CAPS query and words with scrambled case ("WoRkshop") before the encoder and reranker
RETRIEVAL_QUERY_FIX_CASE=true
# Remove Arabic diacritics (harakat) from the query before the dense encoder and the reranker
RETRIEVAL_QUERY_STRIP_HARAKAT=true
# Add the chunks before and after each hit to its text (small-to-big context)
RETRIEVAL_NEIGHBOURS=false
# With a query rewrite: each translation searches only the chunks of its language
RETRIEVAL_REWRITE_LANG_FILTER=true
# With a query rewrite, the reranker's query per chunk: chunk_lang (translation in the chunk's
# language), query (standalone rewrite), original (as typed) or en (English-only rerankers)
RETRIEVAL_REWRITE_RERANK=chunk_lang
# Rank multiplier (0..1] for chunks flagged at ingest as instructions to an AI (prompt injection); 1.0 = off
RETRIEVAL_INSTRUCTION_PENALTY=1.0
# Optional LLM judge for the e2e eval (eval/e2e/judge.py): any OpenAI-compatible endpoint; it never uses OPENROUTER_API_KEY
EVAL_JUDGE_BASE_URL=
# Model name at that endpoint; the judge is off unless both base URL and model are set
EVAL_JUDGE_MODEL=
# API key for the judge endpoint (blank for a local server)
EVAL_JUDGE_API_KEY=
# Per-call judge timeout in seconds (default 60); a failed or timed-out verdict is a skip
EVAL_JUDGE_TIMEOUT=
# Public host name for HTTPS (docker-compose.https.yml); must point at this machine
DOMAIN=
```
