# Requirements Document

## Introduction

The data-preprocessing phase turns raw club material into clean, attributed text that the
rest of the `mcclub_rag` pipeline (chunking → embedding → Qdrant) can consume. It covers three
entry points: **file parsing** (PDF, DOCX, XLSX, MD, TXT) via xberg, **OCR** of images and
scanned pages via Tesseract (`ara`, `fra`, `eng`), and **web page ingestion** via httpx +
trafilatura. Every entry point produces the same output contract — a `ParsedDocument` — so
the downstream chunker never needs to know where text came from.

The corpus is multilingual (Arabic, French, English, frequently mixed), runs fully offline
inside the container on CPU, and has to be robust to the messy inputs a club actually has:
scanned flyers, phone photos of notices, spreadsheets of schedules, and arbitrary URLs.
Chunking, embedding, storage and the HTTP endpoints are **out of scope**; they are only
referenced where this phase hands data to them.

## Requirements

### Requirement 1 — Unified output contract

**User Story:** As the developer of the chunking/indexing stage, I want every source to
produce one well-defined document object, so that downstream code has a single input type.

#### Acceptance Criteria

1. WHEN any supported source is processed successfully THEN the system SHALL return a
   `ParsedDocument` containing: `content_hash`, `title`, `source` (filename or final URL),
   `source_type` (`file` | `web`), `mime_type`, `visibility`, `language`, `text`, an ordered
   list of `sections` (each with text and, where known, `page` number and `heading`),
   `ocr_used` flag, `parser` name/version, and `created_at`.
2. WHEN the same bytes (file) or the same extracted text (web) are processed twice THEN the
   system SHALL produce the same `content_hash` (SHA-256), so the pipeline can detect duplicates.
   The stable `doc_id` is NOT assigned here: it is derived from the document's `topic` by
   `ingest/pipeline.py` (agreed boundary with the API side), so a re-uploaded topic keeps its id.
3. WHEN a caller supplies a `visibility` of `public` or `members` THEN the system SHALL carry it
   unchanged onto the `ParsedDocument`; IF any other value is supplied THEN the system SHALL
   reject the input with a validation error.
4. WHEN a title is not supplied by the caller THEN the system SHALL derive one from document
   metadata, then the first heading, then the filename / page `<title>`, in that order.

### Requirement 2 — File parsing

**User Story:** As a club admin, I want to upload PDF, DOCX, XLSX, Markdown and plain-text
files, so that their content becomes answerable by the chatbot.

#### Acceptance Criteria

1. WHEN a PDF with a text layer is parsed THEN the system SHALL extract its text page by page
   and record the page number on each section.
2. WHEN a DOCX is parsed THEN the system SHALL preserve heading structure as section
   boundaries and include table contents as text.
3. WHEN an XLSX is parsed THEN the system SHALL extract every non-empty sheet, emit one section
   per sheet named after the sheet, and render rows so that header→value associations survive
   (e.g. Markdown table or `header: value` lines).
4. WHEN a MD or TXT file is parsed THEN the system SHALL decode it as UTF-8, falling back to
   charset detection IF UTF-8 decoding fails, and SHALL treat Markdown headings as section
   boundaries.
5. IF the file's detected type (magic bytes) is not in the supported set THEN the system SHALL
   raise an `UnsupportedFileType` error naming the detected type, regardless of the extension.
6. IF the file is password-protected, corrupted, or cannot be opened THEN the system SHALL
   raise a `ParseError` with a human-readable reason and SHALL NOT crash the process.
7. IF the file exceeds the configured max upload size THEN the system SHALL reject it before
   parsing with a `FileTooLarge` error.

### Requirement 3 — OCR

**User Story:** As a club admin, I want scanned PDFs and photos of notices to be readable, so
that information that only exists on paper still reaches the chatbot.

#### Acceptance Criteria

1. WHEN an image file (PNG, JPEG, TIFF, WEBP) is ingested THEN the system SHALL run Tesseract
   OCR with languages `ara+fra+eng` and set `ocr_used = true`.
2. WHEN a PDF page has no extractable text layer (or text below a configurable character
   threshold) THEN the system SHALL OCR that page only, and keep native text for the others.
3. WHEN OCR is run THEN the system SHALL use the traineddata under `TESSDATA_PREFIX` and SHALL
   NOT make any network request.
4. IF the `ara`, `fra` or `eng` traineddata is missing at startup THEN the system SHALL fail fast
   with a clear configuration error rather than silently OCR-ing with fewer languages.
5. WHEN OCR output contains Arabic THEN the system SHALL return text in logical (reading) order
   suitable for embedding, not visual order.
6. IF OCR produces no meaningful text (empty or below the threshold) THEN the system SHALL raise
   an `EmptyDocument` error instead of indexing noise.

### Requirement 4 — Web page ingestion

**User Story:** As a club admin, I want to submit a URL (e.g. a club page or announcement), so
that its main content is ingested without menus, footers and cookie banners.

#### Acceptance Criteria

1. WHEN a URL is submitted THEN the system SHALL fetch it with httpx using a configurable
   timeout, a bounded number of redirects, and an identifying User-Agent.
2. WHEN the response is HTML THEN the system SHALL extract main content with trafilatura
   (boilerplate removed, headings and tables kept) and use the page title / metadata for
   `title`, and the final URL after redirects as `source`.
3. WHEN the response content-type is a supported document type (e.g. `application/pdf`) THEN the
   system SHALL route the bytes through the file parser (Requirement 2/3) with
   `source_type = web`.
4. IF the response status is not 2xx, the request times out, or the body exceeds the
   configured max size THEN the system SHALL raise a `FetchError` stating the reason.
5. IF the URL scheme is not `http`/`https`, or the host resolves to a private, loopback,
   link-local or otherwise internal address (including after redirects) THEN the system SHALL
   refuse the request (SSRF protection).
6. IF trafilatura extracts no main content THEN the system SHALL raise `EmptyDocument`.
7. The system SHALL ingest only the single submitted URL; crawling/link-following is out of
   scope.

### Requirement 5 — Text normalization

**User Story:** As the developer of retrieval, I want consistently cleaned text, so that dense
and BM25 matching aren't defeated by invisible characters or formatting noise.

#### Acceptance Criteria

1. WHEN text is produced by any parser THEN the system SHALL apply Unicode NFC normalization,
   remove control and zero-width characters (except those needed for correct Arabic/RTL
   rendering), unify line endings, and collapse runs of whitespace while preserving paragraph
   breaks.
2. WHEN text contains words hyphenated across line breaks (common in PDFs) THEN the system SHALL
   re-join them.
3. WHEN text contains Arabic THEN the system SHALL remove tatweel (ـ) and SHALL keep the stored
   text otherwise faithful to the source (no destructive folding of letters/diacritics in the
   stored text).
4. WHEN a page header/footer line repeats on most pages of a PDF THEN the system SHALL drop it.
5. Normalization SHALL be a pure, deterministic function, testable in isolation.

### Requirement 6 — Language detection

**User Story:** As the developer of retrieval and evaluation, I want each document tagged with
its language, so that we can analyse quality per language (notably the Arabic subset).

#### Acceptance Criteria

1. WHEN a document is produced THEN the system SHALL detect its dominant language with
   py3langid and store an ISO 639-1 code in `language`.
2. WHEN a document contains substantial content in more than one of `ar`/`fr`/`en` THEN the
   system SHALL record the language per section as well, and SHALL mark the document `mixed`.
3. IF the text is too short for reliable detection THEN the system SHALL set `language = "und"`.
4. WHEN `detect_language(text)` is called on a chat message THEN the system SHALL always return one of
   `"ar"`, `"fr"`, `"en"` (never `und`/`mixed`), so the answer layer can pick a reply language:
   IF Arabic-script letters make up the majority of letters THEN it SHALL return `"ar"`; otherwise the
   most probable of the three; IF the text contains no letters THEN it SHALL return the configured
   default language.

### Requirement 7 — Non-functional constraints

**User Story:** As the operator, I want preprocessing to be safe and predictable on a small
CPU-only container, so that one bad upload can't take the service down.

#### Acceptance Criteria

1. The system SHALL run with no network access other than the explicit URL fetch in
   Requirement 4 (`HF_HUB_OFFLINE=1`).
2. The system SHALL use no RAG/agent framework (plain Python + xberg, trafilatura, httpx,
   py3langid).
3. WHEN parsing a document THEN the system SHALL enforce a per-document time limit and a
   maximum page count, and SHALL guard against decompression bombs in DOCX/XLSX archives.
4. Blocking parse/OCR work SHALL be callable from async FastAPI code without blocking the event
   loop (run in a worker thread/process).
5. WHEN a document is processed THEN the system SHALL emit a structured log line (structlog)
   with `content_hash`, source, type, page count, `ocr_used`, language, and duration — and SHALL
   NOT log document text.
6. All limits (max size, max pages, timeouts, OCR threshold) SHALL be configurable through
   pydantic-settings with sane defaults.

### Requirement 8 — Testability

**User Story:** As a developer, I want the phase covered by automated tests, so that I can
change parsers or settings without regressions.

#### Acceptance Criteria

1. The repository SHALL contain small fixture files under `tests/fixtures/` for each supported
   type, including a scanned (image-only) PDF and an Arabic sample.
2. WHEN `pytest` runs THEN tests for parsing, OCR, normalization and language detection SHALL
   pass without network access; web tests SHALL use a mocked HTTP transport.
3. Every error type in this document SHALL have at least one test that triggers it.

## Open questions

1. **Max sizes / limits:** proposed defaults — 25 MB per file, 300 pages, 60 s per document if it's frozen,
   5 MB per web response. OK
2. **Image formats:** is PNG/JPEG/TIFF/WEBP enough
3. **Arabic folding for BM25:** I kept stored text faithful (Req 5.3). Folding alef variants /
   taa marbuta / diacritics *for the sparse index only* would help Arabic BM25 recall, but it
   belongs to the indexing stage. Should I expose a `fold_for_search()` helper here anyway yes
