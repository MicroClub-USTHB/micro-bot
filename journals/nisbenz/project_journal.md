# Project journal: nisbenz

2026-10-09

2026-10-10

- Retrieval spec'd and built. Measured the rerankers before choosing: gte is 2.3x faster than bge
  on CPU, and both int8 models change scores a lot inside a batch, so one pair per run.
- Eval set of 48 questions with verbatim evidence. The small corpus saturates hit@5, so MRR
  decided the defaults: top_k 20, RRF k 2, gate 0.40 for gte.
- To raise with eequaled: error class mismatch in the app handler, config defaults, ownership.

2026-10-11

- Took over the ingestion side end to end. Every document was titled "Metadata": the corpus files
  are parse.py dumps and were ingested with their metadata block. Fixed in the eval loader.
- Ran the chunk size comparison that was never done: 300 tokens with the heading header is best;
  the header matters more than the size.

2026-10-12

- Spec'd fixes for all 24 issues the e2e eval found, then built my side. The follow-up bug was
  the easiest win: "Who is the dean?" was glued to the hackathon question just for being short.
- The gate cannot be fixed by tuning: answerable and unanswerable scores overlap. It becomes a
  relevance floor, lowered once the LLM is wired.
- Cross-lingual retrieval improved a little (0.755 -> 0.776, 0.816 with bge-m3) but not to the
  target; gte on darija is the next lever.
- OCR on phone photos is blocked: xberg reports the same confidence for a bad photo as for a clean
  scan.
- Then fixed the rest, eequaled's side included, as the user asked: the app finally calls the
  LLM, and without it the user gets a short notice and sources instead of pasted chunks. The
  photo OCR problem turned out to be skew alone: straightening took it from 41% to 99%.

2026-10-13 (evening)

- Read the report back as a to-do list: leaks first, then the cheap retrieval misses.
- The prompt leak was plain to catch: the model copies the prompt sentence by sentence, so a string
  check against the instructions is enough. Ran it over all 1,151 recorded replies before
  trusting it: it found exactly the 7 leaks I knew.
- Wikipedia: a quarter of the corpus was link URLs. Dropping them from what the encoders read
  fixed the targeted questions but did not move recall overall; I kept it for the 10% faster
  rerank. Two of the four USTHB misses are candidate depth, not noise.
- Prompt rules for refusals: the first version made the model answer "NOT_FOUND… no,
  non-students cannot join". It read "the exact fact asked" too literally on yes/no questions.
  One more sentence fixed it. Lesson: run the whole set with only the prompt changed, or every
  number is mixed with the corpus and code changes.
- Found a scorer bug too: a correct refusal said "you should vote for", which was a must_not.

2026-10-15

- The organizers allowed a second LLM call, so darija, arabizi and typos became one problem:
  have the LLM rewrite the question (standalone, corrected, translated into fr / en / ar) before
  searching. No arabizi detector needed: every question goes through it.
- Kept the typed query in the search on purpose. The rewrite only adds arms, so a bad or missing
  rewrite can't lose what we had. That mattered the same day: the free gemma endpoint answered
  429 to every call, and nemotron took 20 s until reasoning was turned off.
- Results: darija hit@5 0.66 -> 1.0, cross-lingual 0.81 -> 0.98, typos mostly gone. Answers
  85% fully correct by the rules, darija 54% -> 83%.
- I expected clean queries to let us shrink both models. Half true: the reranker shrank (mmarco,
  5x smaller, same hit@5), the embedder did not. Small embedders are fine with a rewrite and bad
  without one, and the fallback is exactly when we need it.
- The eval reports were hard to act on: 42 pass/fail slices and a 600-line dump. Rebuilt them
  around "where did it fail": retrieval, refusal, wrong fact, citation. Found the 84% false-answer
  rate was the open gate on the retrieval layer, not the chatbot.
- The biggest leftover is not retrieval: right answers whose snippet shows the wrong sentence,
  because the snippet is matched on the question's words. Next fix.
- The local judge is not worth much: it calls the app's own [Title] citations unsupported.
