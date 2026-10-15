# Design: Query rewrite

## Flow

```
chat: message -> rewrite (LLM call 1, optional) -> retrieve(message, role, history, rewrite=...)
      -> answer (LLM call 2)
```

`llm/rewrite.py` holds the prompt, the parser and `QueryRewriter` (cache + timeout). The chat
route decides whether to call it (breaker, quota with 2 free slots) and adds its usage to the
answer's. `RetrievalBackend.retrieve` takes `rewrite: QueryRewrite | None = None`; every fake
accepts it.

## Prompt

System: the role (search-query rewriter for MC Club / USTHB documents), the JSON shape, the
rules of Req 1.3, one darija and one arabizi example. User: `Previous question: ...` (only with
history) and `Question: ...`, each in a fenced block. Short: about 300 prompt tokens, 80-120
completion tokens.

`parse_rewrite` takes the first `{...}` block of the reply (models wrap JSON in fences), checks
types and lengths, drops empty or duplicate translations, and maps an unknown `lang` to `None`.

## Retrieval

```
arms = dense(original) + BM25(original + rewrite texts)
     + dense(query) [all chunks] + dense(fr) [fr chunks] + dense(en) [en] + dense(ar) [ar]
```

All arms go into one `query_points` with RRF (weights: dense arms `dense_weight`, BM25
`sparse_weight`). Texts equal to the original after normalization are not encoded twice.
Rerank: candidates are grouped by their query text (translation in the chunk's `language`,
else `query`), each group scored with `Reranker.score(query, passages)`, so the protocol does
not change. `Candidate` gets `language` from the payload.

Without a rewrite, the code path is the one before this spec (cross-language arm included).

## Encoders

`EmbedderSpec` gets `pooling: "cls" | "mean"`, `query_prefix`, `passage_prefix` (default
empty). `OnnxDenseEncoder` applies them; the prefixes are in the fingerprint, so the index
signature changes when they do. ModernBERT models (granite r2) take no `token_type_ids`; e5
(XLM-R) does, fed with zeros when the session asks for it.

## Eval

- Rewrites in the eval go through `RecordingGenerator` (same store, keyed by model + prompt),
  so they are recorded once (ollama `gemma4-12b-eval` locally, or OpenRouter) and replayed.
- `eval/bench_models.py`: per embedder, ingest once into in-memory Qdrant; per question,
  candidates with and without the rewrite; each reranker scores the same candidates. One JSON +
  markdown under `eval/results/models/`.
- Gate analysis: positives = answerable questions whose evidence is in the reranked top 5;
  negatives = unanswerable questions plus answerable ones with no evidence in the top 5. ROC AUC
  of the top score, best Youden threshold, and the false-refusal / caught-miss rates at it.
- E2E: the in-process driver keeps the last `Retrieval` of each chat request, so a chat result
  knows whether the evidence was in the passages the LLM saw. Attribution per failed answerable
  case: `retrieval_miss` (evidence not retrieved), `generation_refused` (retrieved, refused),
  `generation_wrong` (retrieved, answered, gold missing), `citation_miss` (answered with gold,
  evidence not cited), `unfaithful` (judge). Language groups: `arabizi` and `darija` slices
  first, else the case language.
