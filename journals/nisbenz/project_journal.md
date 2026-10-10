# Project journal: nisbenz

2026-10-09

2026-10-10

- Retrieval spec'd and built. Measured the rerankers before choosing: gte is 2.3x faster than bge
  on CPU, and both int8 models change scores a lot inside a batch, so one pair per run.
- Eval set of 48 questions with verbatim evidence. The small corpus saturates hit@5, so MRR
  decided the defaults: top_k 20, RRF k 2, gate 0.40 for gte.
- To raise with eequaled: error class mismatch in the app handler, config defaults, ownership.
