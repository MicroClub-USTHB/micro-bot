# 4 GB / 2-core simulation: results

What MC-RAG does on a 4 GB / 2-core VPS, measured against the real application. Summary table on the
front page: [README.md](../../../README.md#deployment--the-whole-stack-on-a-4-gb--2-core-vps-2026-10-15).
Runbook and the staged procedure: [`docs/simulation/4gb-vps.md`](../../../docs/simulation/4gb-vps.md).
Spec: [`specs/vps-simulation/`](../../../specs/vps-simulation/requirements.md).

**Status: run on 2026-10-15, at the real 2 CPU / 3.83 GB ceiling, scenario S0 (retrieval only).**
Every row below is either *measured* with the command that produced it, or explicitly *not
measured*. No load number is estimated.

Raw samples (`<UTC>-<label>-stats.csv`) and eval runs (`../e2e/`) stay local, like every other eval
result.

---

## 1. Host and toolchain (measured, 2026-10-15)

| Item | Value |
|---|---|
| Host | Windows, 12 logical cores, 15.92 GB RAM |
| Docker Desktop | CLI 29.6.2, WSL 2 kernel `6.18.33.2-microsoft-standard-WSL2` |
| Compose | 5.6.0 (`!override` needs >= 2.24) |
| Python | 3.12 via `uv` |
| **VM before the simulation** | **12 CPUs, 7.72 GB** (8,284,422,144 bytes) - no `.wslconfig` |
| **VM after `-Mode Apply`** | **2 CPUs, 3.83 GB** (4,107,382,784 bytes), swap 0 - verified with `docker info` |

The gap between those last two rows is why the simulation exists: a benchmark run in the default VM
would overstate a 4 GB / 2-core VPS by roughly 2x memory and 6x CPU.

## 2. What the containers are actually given (measured)

From `docker compose -f docker-compose.yml -f docker-compose.simulation.yml config`, cross-checked
by `tests/test_simulation_compose.py`:

| Property | app | qdrant |
|---|---:|---:|
| Memory limit | 2,936,012,800 B (2800 MB) | 536,870,912 B (512 MB) |
| Memory reservation | 1024 MB | 256 MB |
| Memory + swap | = limit (no extra swap) | = limit |
| CPU quota | 1.5 | 0.5 |
| PIDs | 512 | 256 |
| Published port | `127.0.0.1:8001 -> 8000` (only) | none (`expose 6333`, internal network) |

Project `mc-rag-sim` -> network `mc-rag-sim_default`, volumes `mc-rag-sim_app_data` and
`mc-rag-sim_qdrant_storage`. The normal deployment is a different project and is never named.
Sum: 3312 MB of the 4 GB VM; the remaining ~0.7 GB is Linux, Docker and the page cache.

## 3. Harness verification (measured)

| Check | Result |
|---|---|
| Default offered rate unchanged | `interval == 3.0 s` (20 per 60 s) |
| Custom rate applied | 240/min -> 0.25 s |
| Non-positive rate rejected | `--requests-per-minute 0` and `-5` exit with a usage error |
| Concurrency safety | 8 concurrent tasks: **7** shared one pacing slot before the lock, **1** after |
| Missing host RSS | `peak_rss_mb: null`, report prints `n/a` (Windows has no `getrusage`) |
| Secrets in script output | the resolved config contains the key; `Protect-SimText` removes it |

## 4. Stages: what has and has not run

| Stage | Status |
|---|---|
| 0 - VM limits | **applied and verified**: 2 CPUs / 3.83 GB, swap 0 |
| 1 - preflight | **run, passed** (exit 0) |
| 2 - build | **run**: `mc-rag-sim-app:latest`, 2.2 GB, all models baked in (embedder 559 MB, reranker 130 MB, chunk tokenizer 32 MB, tessdata ara/eng/fra) |
| 3 - start / health | **run**: app healthy on `127.0.0.1:8001` only, Qdrant with no host port |
| 4 - retrieval smoke | **run**: upload -> `documents: 1` -> answer with 1 source, correct title, p50 ~600 ms; survived a VM restart (volume persists) |
| 5 - Profile A (shipped rate limit) | **run**: 19 req at 18/min, p50 146.5 ms, p95 169.0 ms, 0 errors |
| 6 - Profile B ramp (1x30 s, 2x60 s, 4x120 s, 8x120 s) | **run**: saturates at 5.19 req/s (see section 5) |
| 7 - ingestion | **not run** (corpus upload stage untouched; the smoke used one synthetic document) |
| 8 - cleanup | script runs correctly with nothing up (exit 0) |

## 5. Ramp results (measured, S0, Profile B)

Retrieval only, no provider call, one synthetic document in the store.

| Run | Concurrency | Offered /min | Requests | Achieved req/s | p50 | p95 | p99 | errors | 429s | app peak MB | qdrant peak MB | app CPU | OOM | restarts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| simB-c1-60rpm | 1 | 60 | 31 | 1.03 | 152 ms | 188 ms | 197 ms | 0 | 0 | 1895 | 75 | 34% | no | 0 |
| simB-c2-120rpm | 2 | 120 | 122 | 2.01 | 150 ms | 242 ms | 374 ms | 0 | 0 | 1895 | 75 | 124% | no | 0 |
| simB-c4-240rpm | 4 | 240 | 484 | 4.00 | 147 ms | 391 ms | 702 ms | 0 | 0 | 1904 | 75 | 150% | no | 0 |
| simB-c8-480rpm | 8 | 480 | 625 | **5.19** | **1520 ms** | 1957 ms | 2632 ms | 0 | 0 | 1913 | 75 | 151% | no | 0 |

Profile A (the shipped 20 req / 60 s limit), 18/min for 60 s: 19 requests, 0.32 req/s, p50 146.5 ms,
p95 169.0 ms, p99 182.4 ms, 0 errors - the limit, not the box, is the ceiling there.

Two readings of the same table:

- **Throughput scales linearly to concurrency 4** (1.03 -> 2.01 -> 4.00 req/s) with p50 flat at
  ~150 ms: every additional concurrent user gets its own ~150 ms slot, nothing queues.
- **The knee is between 4 and 8.** At concurrency 8 throughput stops at 5.19 req/s (about 310
  requests/minute) and p50 rises 10x to 1.5 s. The app is pinned at **150% CPU, its whole 1.5-CPU
  quota**, from concurrency 4 onward, so the limiter is **CPU, not memory**.

## 5b. Generation (S1) could not be measured

S1 was attempted first and degraded to the extractive fallback on every request
(`grounded: false`, `usage.llm_calls` counted, `prompt_tokens` and `completion_tokens` 0). A probe
made from inside the app container against OpenRouter returned:

```
HTTP 429 {"error":{"message":"Provider returned error","code":429,
"metadata":{"raw":"google/gemma-4-31b-it:free is temporarily rate-limited upstream."}}}
```

The free tier's shared upstream pool was rate-limited for the whole window, so **no generation
latency or generation-time CPU number is reported here**. The ramp above is S0 by necessity and by
design: it isolates retrieval capacity, which is what the 2-core ceiling actually constrains.
Retrying S1 later only needs the key uncommented in `.env` and
`docker compose up -d --force-recreate app`; the harness is unchanged.

## 6. Verdict

**Fits, with one caveat.** On the 2 CPU / 3.83 GB VM, with 3312 MB of it handed to the two
containers, MC-RAG serves retrieval-only traffic comfortably at **4 req/s with p50 ~150 ms** and
saturates at **~5.2 req/s (about 310 requests/minute)**, where p50 degrades to 1.5 s. The first
component to saturate is **the app's CPU quota** (150% of 150%, from concurrency 4 on), not RAM:
memory held flat at 1.83-1.91 GB with no leak, Qdrant stayed at 75 MB and 1.6% CPU, and no run
produced an OOM kill, a restart, an error or a rate limit.

The caveat is S1: generation could not be exercised (section 5b), so these numbers describe the
retrieval path. Since generation is I/O-bound on the provider, the CPU ceiling found here is an
upper bound for what an S1 run would spend on the box.

Next test: stage 7 (ingestion under the same ceiling), and an S1 retry when the free pool recovers.

## Reproducing

```powershell
powershell -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Show     # then Apply, restart, re-check
powershell -File scripts/simulation/Test-SimPreflight.ps1
powershell -File scripts/simulation/Start-Simulation.ps1 -Build
powershell -File scripts/simulation/Invoke-SimBenchmark.ps1 -Scenario S0 -Profile B -Concurrency 4 -Duration 120 -RequestsPerMinute 240 -Label simB-c4-240rpm
powershell -File scripts/simulation/Stop-Simulation.ps1
```