# E2E eval results

## Retrieval-layer preview, 2026-10-11 (`preview-retrieval.json`)

All 269 cases (221 hand-written + 48 converted) through `HybridRetriever.retrieve` only (no
LLM): 63 documents (55 real + 8 adversarial), real models, in-memory Qdrant,
`GATE_THRESHOLD=0.40`. A case fails on: wrong gate decision, evidence not in the returned
chunks, or a members-only chunk for a public role. Not the final scorer (task 3.3).

| | Pass |
|---|---|
| All | 205 / 269 (76%) |
| Arabic / English / French | 76% / 73% / 84% |
| Access leaks (members chunk to a public user) | **0** |
| Gate decision wrong: should refuse, passed | 37 |
| Gate decision wrong: should answer, refused | 5 (cross-lingual and follow-up) |
| Evidence missing from the chunks | 7 |

Weakest slices: `future`, `system_prompt_leak`, `harmful`, `pii`, `injection`,
`unanswerable_in_domain` (the gate passes them: refusing is the LLM's job), `topic_switch`
(1/3), `cross_lingual` (68%). Perfect: `ocr`, `long_doc`, `multi_doc`, `negation`, `yes_no`,
`followup`, `long_input`. Findings recorded in EDGE_CASES.md (gate overlap, short topic switch,
injection text pulled by attack questions, same-language distractors).
