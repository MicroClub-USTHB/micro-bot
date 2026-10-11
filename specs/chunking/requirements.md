# Requirements Document

## Introduction

Chunking turns one `ParsedDocument` (output of `specs/data-preprocessing/`) into an ordered list of
retrieval units (`Chunk`s). The pipeline (`ingest/pipeline.py`) then embeds them and stores them in
Qdrant. Chunking is where most retrieval failures start: a chunk that holds half a fact, or one with no
idea which section it came from, can't be found by dense or BM25 search and can't be used by the LLM.

The strategy is **structure first, sentences second, size last**, and it is chosen from published
results:

- **Respect structure.** Split at headings first. Only split *inside* a section when it is too large,
  and then at paragraph and then sentence boundaries. A chunk never ends mid-sentence unless one
  sentence is longer than the token cap. Sentence-aware chunking scored best for Arabic RAG
  (74.8 vs 69.4 fixed-size, 66.9 semantic; arXiv 2506.06339) and tied semantic chunking at a fraction
  of the cost on NQ (arXiv 2601.14123).
- **No embedding-based "semantic" chunking.** NAACL 2025 (Vectara, arXiv 2410.13070) found the cost is
  not justified by consistent gains, and in a 2026 benchmark it produced tiny fragments (about 43
  tokens) that dropped end-to-end accuracy.
- **Medium chunks, about 80 to 400 tokens (target 300).** The optimum reported across studies is
  150 to 512 tokens. Our cap also has to leave room for the query inside the reranker's 512-token window
  (bge-reranker-v2-m3).
- **No overlap.** A 2026 ablation (arXiv 2601.14123) found 10 to 20% overlap gave no measurable gain
  and made the index 1.11× to 1.25× larger. Lost context is restored in two other ways: a heading
  breadcrumb on every chunk, and neighbor links so retrieval can expand to adjacent chunks.
- **Deterministic contextual headers, not LLM-written ones.** Anthropic's Contextual Retrieval cut
  retrieval failures by 49% (67% with reranking) by prepending context to each chunk before embedding
  *and* BM25 indexing. We can't spend LLM calls at ingest (50 requests a day, offline container), so
  we prepend the document title plus the heading path instead.

The real corpus (`data/parsed/`) shows what has to be handled: docs-site pages with clean H1 to H3
trees, Arabic articles with long H2 sections, a news homepage with many short sections, and a
slide-style PDF that xberg turns into dozens of tiny "sections" (`## 01`, `## 1`, 5 to 60 characters).

**In scope:** `ingest/chunk.py` (pure, deterministic `chunk_document()`), the `Chunk` model, the
chunking settings, and the tests. **Out of scope:** embedding, Qdrant, `doc_id`/topic, neighbor
expansion at query time (retrieval spec), and changes to preprocessing. The chunker reads
`ParsedDocument` as it is.

## Requirements

### Requirement 1: Chunk contract

**User Story:** As the ingest pipeline, I want one well-defined `Chunk` type per retrieval unit, so
that embedding, BM25 indexing, payload storage and source citations all read the same fields.

#### Acceptance Criteria

1. WHEN `chunk_document(doc)` is called with a `ParsedDocument` THEN the system SHALL return an ordered
   tuple of frozen `Chunk`s, each with: `index` (0-based, contiguous), `text` (the faithful chunk
   text shown to the LLM and used for snippets), `embed_text` (the text that is embedded and
   BM25-indexed), `heading_path` (tuple of ancestor headings, outermost first), `page_start` and
   `page_end` (or `None`), `language`, and `token_count` (the token count of `embed_text`).
2. WHEN the same `ParsedDocument` is chunked twice THEN the system SHALL return identical chunks
   (pure function: no clock, randomness, I/O or global state besides the loaded tokenizer).
3. WHEN chunks are concatenated in `index` order THEN every non-whitespace character of the
   document's sections SHALL appear exactly once, apart from the section-heading lines that move
   into `heading_path` (no content dropped, no content duplicated).
4. IF a document has no sections with alphanumeric content THEN the system SHALL return an empty
   tuple, and the pipeline decides how to report it.
5. The chunker SHALL NOT assign chunk ids. The pipeline derives them (for example
   `uuid5(doc_id, str(index))`).

### Requirement 2: Structure-first boundaries

**User Story:** As a club member asking a question, I want each chunk to cover one coherent part of a
document, so that the retrieved text holds the whole answer and not half of it.

#### Acceptance Criteria

1. WHEN a document has Markdown headings THEN the system SHALL treat every heading as a chunk
   boundary, and SHALL NOT put text from two different sections into one chunk unless
   Requirement 4 (merging small sections) applies.
2. WHEN a section's heading line starts with `#`×N THEN the system SHALL rebuild the heading
   hierarchy from those levels. The heading path of a section is its own heading plus the nearest
   preceding heading of each lower level.
3. WHEN a PDF section has no heading line of its own (it continues a heading from an earlier page)
   THEN the system SHALL keep the heading path of the section it continues.
4. WHEN a section spans several pages, or a chunk merges sections from several pages, THEN
   `page_start`/`page_end` SHALL cover every page the chunk's text came from.
5. WHEN an XLSX sheet section is chunked THEN the heading path SHALL be the sheet name, and the
   system SHALL split only between row lines (each row line is atomic).

### Requirement 3: Splitting oversized sections

**User Story:** As the retrieval stage, I want no chunk to exceed the model windows, so that nothing
gets silently truncated by the embedder or the reranker.

#### Acceptance Criteria

1. WHEN a section's `embed_text` exceeds `max_tokens` THEN the system SHALL split it, trying
   boundaries in this order: blank-line paragraphs, then single lines, then sentences, then words.
   Each split uses the coarsest level that produces pieces within `max_tokens`.
2. WHEN splitting a section THEN the system SHALL pack consecutive units greedily up to
   `target_tokens` and SHALL NOT exceed `max_tokens`. IF the final piece is smaller than
   `min_tokens` THEN the system SHALL rebalance it with the previous piece, provided the result
   stays under `max_tokens`.
3. WHEN sentences are split THEN the system SHALL recognise Latin and Arabic terminators (`.`, `!`,
   `?`, `…`, `؟`, `؛`, `۔`), including when followed by closing quotes or brackets, and SHALL NOT
   split on decimals (`3.5`), common abbreviations (`e.g.`, `M.`, `Dr.`), or dotted
   versions and URLs.
4. WHEN a block is a fenced code block, a Markdown table, or a list THEN the system SHALL keep it
   whole in one chunk. IF it alone exceeds `max_tokens` THEN the system SHALL split it at line or
   row boundaries, and for a table SHALL repeat the header row (and separator) at the top of every
   piece.
5. IF one sentence or line alone exceeds `max_tokens` THEN the system SHALL split it at word
   boundaries (or at token boundaries if there is no whitespace) and SHALL log a `chunk_hard_split`
   warning with the document title and chunk index, but not the text.
6. Every produced chunk SHALL satisfy `token_count <= max_tokens`.

### Requirement 4: Merging small sections

**User Story:** As a club member, I want slide-style PDFs and short web sections to still produce
useful chunks, so that answers don't come from 5-character fragments like `## 01`.

#### Acceptance Criteria

1. WHEN a section's body is smaller than `min_tokens` THEN the system SHALL merge it with the
   following sibling or descendant sections (same parent heading) while the next section would
   not push it over `target_tokens`. A following section that is itself at least `min_tokens`
   SHALL join only while the merged chunk is still under `min_tokens`, so that runs of tiny
   sections become one chunk instead of several minimum-sized ones.
2. WHEN sections are merged THEN their heading lines SHALL stay inline in `text` (for example
   `## 01` followed by its body) so the structure stays readable. The chunk's `heading_path` SHALL
   be the common parent path of the merged sections.
3. The system SHALL NOT merge across a boundary of a higher level than the merged sections'
   parent (for example two `##` sections under different `#` headings).
4. IF a small section has no following sibling to merge with THEN the system SHALL merge it into the
   previous chunk when that chunk stays within `max_tokens`. Otherwise it SHALL emit it as its own
   (small) chunk.
5. WHEN a section's text is only a heading (no body) THEN the system SHALL NOT emit it as its own
   chunk. IF the next section is its descendant THEN its heading SHALL live on only in the
   descendants' `heading_path`. Otherwise it SHALL be merged forward per 4.1. A trailing
   heading-only section at the end of the document SHALL be appended to the previous chunk.

### Requirement 5: Contextual header for retrieval

**User Story:** As the retriever, I want each chunk to say where it comes from, so that a chunk like
"the deadline is Friday" can be matched to a question about the hackathon registration.

#### Acceptance Criteria

1. WHEN a chunk is built THEN `embed_text` SHALL be a short header followed by a blank line and then
   `text`. The header holds the document title and the heading path joined by ` > `, with
   consecutive duplicate entries removed (for example a title equal to the H1).
2. WHEN `context_header` is disabled in settings THEN `embed_text` SHALL equal `text`, so it can be
   A/B tested.
3. The header SHALL count toward `token_count` and `max_tokens`. IF the header alone exceeds
   `max_header_tokens` THEN the system SHALL truncate it from the left, keeping the innermost
   headings.
4. `text` SHALL NOT contain the header (the LLM prompt and snippets show title and section
   separately from the payload).

### Requirement 6: Token counting

**User Story:** As the developer, I want chunk sizes measured in the real model's tokens and not in
characters, so that Arabic (which tokenizes differently from French and English) isn't over- or
under-chunked.

#### Acceptance Criteria

1. The system SHALL count tokens with the embedder's tokenizer (`tokenizer.json` of `EMBED_MODEL`,
   loaded with `tokenizers`, offline, no special tokens), loaded once and reused.
2. IF the tokenizer file is missing at startup THEN the system SHALL fail fast with an instruction
   naming the expected path (the same pattern as `verify_tessdata()`).
3. Default sizes SHALL be `target_tokens=300`, `max_tokens=400`, `min_tokens=80`,
   `max_header_tokens=48`. These were measured, not guessed: on code-heavy text the reranker's
   tokenizer produces up to 1.28× more tokens than granite's, so a 400-token chunk plus the query
   fits a reranker `max_length` of 640 (see design). All of them SHALL be configurable through
   `INGEST_CHUNK_*` settings, which are validated: `min < target <= max`.
4. The tokenizer SHALL ignore any truncation or padding configured inside `tokenizer.json`
   (granite's ships with truncation at 32,768 tokens and padding enabled).

### Requirement 7: Neighbor links and metadata for retrieval

**User Story:** As the retrieval stage, I want to know each chunk's neighbors and section, so that I
can expand a hit to the surrounding text (small-to-big) without overlap in the index.

#### Acceptance Criteria

1. WHEN chunks are produced THEN each chunk SHALL carry `section_index` (index of the first source
   section) so the pipeline can store it and retrieval can fetch `index ± 1` chunks of the same
   document or the same section.
2. WHEN a chunk's sections have different `language` tags THEN the chunk `language` SHALL be the
   language covering the most characters, falling back to the document language when all are
   `und`.

### Requirement 8: Observability and performance

**User Story:** As an operator, I want to see what chunking did without leaking document content.

#### Acceptance Criteria

1. WHEN a document is chunked THEN the system SHALL log one `document_chunked` event with the chunk
   count, min, median and max `token_count`, and the counts of merges and hard splits. It SHALL NOT
   log any text.
2. WHEN a 300-page text PDF is chunked on CPU THEN chunking SHALL take under 2 s. Tokenization is
   the cost, so the system SHALL batch-encode units and not re-tokenize already-measured text
   repeatedly.

### Requirement 9: Evaluable configuration

**User Story:** As the team, I want to compare chunking variants on `eval/questions.jsonl`, so that
the defaults are backed by our corpus and not only by external benchmarks.

#### Acceptance Criteria

1. WHEN the chunking settings change THEN the produced chunks SHALL change only through those
   settings (no hidden constants), so that `target_tokens` ∈ {200, 300, 400} and `context_header`
   on/off can be evaluated with hit@5 and MRR, including the Arabic subset.
2. The chunking settings SHALL be part of the corpus version (`corpus_version()` in the pipeline
   spec), so that changing them invalidates the answer cache and triggers a reindex.

## Open questions

- Should a document title that differs from the H1 appear in the header twice (title > H1 > H2), or
  only the path? The current draft keeps both and removes them only when they are equal.
- Should repeated boilerplate sections (the BBC homepage "follow us" or "app" blocks) be dropped
  here or in preprocessing? The draft leaves them to preprocessing and chunks them like any
  section.
