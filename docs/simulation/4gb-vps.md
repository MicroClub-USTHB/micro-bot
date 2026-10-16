# Running MC-RAG on a 4 GB / 2-core VPS

A reproducible way to answer three questions without buying a VPS yet:

1. **Does it fit?** 4 GB and 2 vCPUs for the app, Qdrant, the models and the operating system.
2. **How fast is it?** Latency and throughput against the real application, on real models.
3. **What breaks first?** Which component saturates, and where the ceiling really is.

Nothing here changes how MC-RAG behaves in production: the same `Dockerfile`, the same services,
the same models, the same code. The simulation is a separate Compose **project** (`mc-rag-sim`)
whose containers, network and volumes cannot collide with your normal deployment.

- Spec: [`specs/vps-simulation/`](../../specs/vps-simulation/requirements.md) (requirements, design,
  tasks)
- Measured on this machine (16 GB RAM, 12 cores, Windows 11, Docker Desktop, WSL 2 kernel
  `6.18.33.2-microsoft-standard-WSL2`): Docker CLI 29.6.2, Compose 5.6.0, Python 3.12 via `uv`.

---

## The two ceilings, and why they are not the same thing

| Layer | Mechanism | Value | What it bounds |
|---|---|---|---|
| VM | `%UserProfile%\.wslconfig` → WSL 2 | `memory=4GB`, `processors=2`, `swap=0` | everything: both containers, Linux, page cache |
| Container | Compose `mem_limit` / `cpus` | app 2800 MB / 1.5 CPU, Qdrant 512 MB / 0.5 CPU | one container each |

Container limits alone would be a lie: Qdrant and the kernel take their share of the VM too.

**Measured baseline.** With no `.wslconfig`, the Docker Desktop VM silently took **7.72 GB and all 12
cores** — half the host's memory, every core. A benchmark run there would have overstated the VPS by
roughly 2x memory and 6x CPU. That gap is the whole reason for the first step.

The app and Qdrant limits add up to ~3.3 GB, leaving ~0.7 GB for Linux, Docker and the page cache.
That split is a **hypothesis to be measured**, not a proven fit; change it only with a reason.

---

## Files

| File | What it is |
|---|---|
| `docker-compose.simulation.yml` | the simulation overlay: project name, loopback port, env, limits |
| `docker-compose.simulation.capacity.yml` | Profile B only: raises the chat rate limit for capacity runs |
| `scripts/simulation/SimCommon.ps1` | shared helpers (project name, secret redaction, container state) |
| `scripts/simulation/Set-SimVmLimits.ps1` | show / apply / restore the WSL 2 VM limits |
| `scripts/simulation/Test-SimPreflight.ps1` | stage 1: prove the environment can run the benchmark |
| `scripts/simulation/Start-Simulation.ps1` | stage 3: build, start, wait for health, print the limits |
| `scripts/simulation/Invoke-SimBenchmark.ps1` | stages 5-6: one labelled load run + `docker stats` sampling |
| `scripts/simulation/Stop-Simulation.ps1` | stage 8: remove the simulation and only the simulation |
| `docs/simulation/4gb-vps.md` | this file |
| `eval/results/simulation/*.csv` | the samples (git-ignored, like every other eval result) |

Nothing is added to `Dockerfile`, `docker-compose.yml`, `.env.example` or the application code.

---

## Scenarios: S0 (default) and S1

| | S0 — retrieval only | S1 — real generation |
|---|---|---|
| `OPENROUTER_API_KEY` in `.env` | commented out | set |
| `/api/v1/health` | `degraded` (**expected**) | `ok` |
| Answer | extractive fallback, `grounded: false`, `usage.llm_calls: 0`, passages as sources | written by `google/gemma-4-31b-it:free` |
| Quota | none | free tier: `LLM_DAILY_LIMIT 50` − `LLM_RESERVE 5` = **45 calls/day**, at most 2 per question |
| Measures | the retrieval stack and the infrastructure around it | the same, plus provider latency that no self-hosted VPS would have had |

`degraded` in S0 is **not** a failure: `/api/v1/health` also probes the generation provider, and
with no key that probe is false. Documents, listing and retrieval are fully functional.

Switching scenarios means editing one line in `.env` and restarting the app (settings are read once
at startup). S0 is the default for every capacity number in this guide, because a 600 rpm run with a
live provider would exhaust the daily quota in minutes.

---

## Profiles: A (shipped rate limit) and B (capacity)

| | Profile A | Profile B |
|---|---|---|
| Overlays | base + simulation | base + simulation + **capacity** |
| Chat rate limit | 20 requests / 60 s (production) | 1000 / 60 s (test only) |
| Answers | "what does the shipped policy do from one client IP?" | "where does this hardware saturate?" |
| Offered load | 20 rpm or less | 60 → 120 → 240 → 480 rpm |

Every simulated user reaches the server from `127.0.0.1`, so they share one client IP: under
Profile A the app's own limiter answers 429 long before the CPUs are busy. Profile A therefore
cannot model several independent remote clients — that is what it measures, and its 429 count is the
result, not a failure.

Profile B's rate limit is a **deliberate test-only value**. It is not a production-recommended
security configuration, and it does not change `.env` or `.env.example`.

---

## The eight stages

Every command is PowerShell 5.1+ from the repository root. All scripts are read-only or additive
except stage 8.

### Stage 0 — decide the VM limits (once)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Show
```

Expected now: `no .wslconfig: the VM uses Docker Desktop defaults` and `12 cpus, 8284422144 bytes`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Apply
```

It backs up your file first (`%UserProfile%\.wslconfig.sim-backup-<UTC>`, or records that there was
none), then merges only `memory`, `processors` and `swap` into the existing `[wsl2]` section,
keeping every other key and section. It **prints** the next steps instead of running them, because
they stop the running VM:

```powershell
# 1. quit Docker Desktop (tray icon -> Quit)
wsl --shutdown
# 2. start Docker Desktop again, then verify:
powershell -ExecutionPolicy Bypass -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Show
```

Verify, do not assume: `-Mode Show` must now print `2 cpus` and about `4294000000` bytes. To go back:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Set-SimVmLimits.ps1 -Mode Restore
```

Gotcha: Docker Desktop's own Resources sliders write to the same file, so re-check after any change
made in its settings.

**Not on WSL 2?** The helper will say so and point at the Hyper-V path, which is manual and needs an
elevated shell: `Set-VMMemory -VMName <DockerDesktopVM> -DynamicMemoryEnabled $false
-DynamicMemoryMinimumBytes 4GB -DynamicMemoryMaximumBytes 4GB` and `Set-VMProcessor -VMName
<DockerDesktopVM> -Count 2`.

### Stage 1 — preflight

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Test-SimPreflight.ps1
```

Proves, without starting anything: the daemon answers, Compose understands `!override` (needs
2.24+), `.env` exists, both overlay combinations parse, the resolved config has the loopback port
and the limits, and no simulation-only value has leaked into `.env`. Secrets are redacted — the raw
`docker compose config` output contains your key in clear text, so nothing prints it.

Exit codes: `2` no daemon, `3` Compose too old, `4` no `.env`, `5` overlay does not parse, `6`
simulation value in `.env`.

### Stage 2 — build (before the ceiling, on purpose)

```powershell
docker compose -f docker-compose.yml -f docker-compose.simulation.yml build app
docker compose -f docker-compose.yml -f docker-compose.simulation.yml pull qdrant
```

Do this with the VM ceiling still lifted: `uv sync` plus ~700 MB of models (bge-m3 568 MB, mmarco
119 MB, BM25 stopwords, `ara`/`fra`/`eng` tessdata) is the most memory-hungry moment of the whole
exercise. Then apply stage 0. The image is project-scoped (`mc-rag-sim-app`), so your dev stack's
image is untouched.

### Stage 3 — start and initialize

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Start-Simulation.ps1
# first time, or after a model change:
powershell -ExecutionPolicy Bypass -File scripts/simulation/Start-Simulation.ps1 -Build
```

Polls `/api/v1/health` for up to 300 s (the ONNX sessions load on the first call), then prints the
per-container memory limit, memory+swap limit, CPU quota, OOM flag and restart count read back from
`docker inspect` — not from the YAML, so a silently ignored key shows up. If the app never answers
it prints the last 40 log lines and exits 5.

### Stage 4 — retrieval smoke test

```powershell
$env:MEMBER_TOKEN = '<the MEMBER_TOKEN in .env>'
Invoke-RestMethod -Uri http://127.0.0.1:8001/api/v1/health | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/api/v1/chat `
  -Headers @{ Authorization = "Bearer $env:MEMBER_TOKEN" } `
  -ContentType 'application/json' `
  -Body (@{ message = 'What are the membership fees?' } | ConvertTo-Json) | ConvertTo-Json -Depth 4
```

Expect the full JSON contract: `answer`, `conversation_id`, `grounded`, `sources[]`, `usage` and
`latency_ms`. In S0: `grounded: false`, `usage.llm_calls: 0`, the passages in `sources`. If you get
`0 sources`, the corpus is empty — go to stage 7 to ingest, then repeat.

First measurement worth writing down: this single-request latency is the floor everything else is
compared against.

### Stage 5 — Profile A, the shipped rate limit

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile A -Concurrency 1 -Duration 30 -RequestsPerMinute 20 -Label simA-c1
```

Record: p50/p95/p99, `rate_limited` (429s are expected here), errors, peak memory per container,
CPU, OOM flags, restarts.

### Stage 6 — Profile B, the capacity ramp

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Start-Simulation.ps1 -Profile B
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile B -Concurrency 1 -Duration 30  -RequestsPerMinute 60  -Label simB-c1-60rpm
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile B -Concurrency 2 -Duration 60  -RequestsPerMinute 120 -Label simB-c2-120rpm
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile B -Concurrency 4 -Duration 120 -RequestsPerMinute 240 -Label simB-c4-240rpm
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile B -Concurrency 8 -Duration 120 -RequestsPerMinute 480 -Label simB-c8-480rpm
```

One run at a time, labelled. These are **initial settings, not supported capacities**. Stop
escalating if the app is OOM-killed, restarts, or stops answering; the last clean run is your
capacity, not the one after it.

What each number means:

- **Offered rate** is a ceiling, not a target: with a 600 ms p50, 4 concurrent users can only offer
  about 6.7 requests per second.
- **p95 rising faster than throughput** is the queue forming — that is the saturation signal.
- **`peak RSS: n/a`** in the eval report is normal and correct on Windows: it is the *load
  generator's* memory, and Windows has no `getrusage`. The server's memory is the `docker stats`
  column the script prints, not that number.

### Stage 7 — ingestion (separately: it is a different workload)

Parsing, OCR and embedding have a completely different CPU and memory profile from serving, so they
are measured on their own:

```powershell
$env:ADMIN_API_KEY = '<the ADMIN_API_KEY in .env>'
powershell -ExecutionPolicy Bypass -File scripts/simulation/Invoke-SimBenchmark.ps1 `
  -Scenario S0 -Profile A -Concurrency 1 -Duration 30 -RequestsPerMinute 60 -Label sim-ingest -Upload
```

`-Upload` loads the eval corpus (`eval/corpus.json` sources under `data/parsed`) through the
documents API. Watch for: time per document, whether the app stays responsive during OCR
(`INGEST_MAX_CONCURRENCY=1` in the simulation), and peak memory — OCR is the most likely source of
an OOM kill in this whole exercise. The typed ingest suite over HTTP is
`uv run python -m eval.e2e.run --target http --base-url http://127.0.0.1:8001 --suite ingest`.

Everything uploaded here stays inside the simulation volume.

### Stage 8 — report, then clean up

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/Stop-Simulation.ps1
powershell -ExecutionPolicy Bypass -File scripts/simulation/Stop-Simulation.ps1 -Force -RemoveImage
```

Asks for confirmation, and refuses any project name other than `mc-rag-sim`. **`down -v` deletes the
simulation's documents and vectors** — they are rebuilt by the next `Start-Simulation`. Your normal
deployment's project is never named by any of these commands.

Fill in the report below. Do not write "it worked" because the containers started.

---

## Report template

| Item | Value |
|---|---|
| VM limits actually in effect | (from `Set-SimVmLimits.ps1 -Mode Show`) |
| Container limits actually in effect | (from `Start-Simulation.ps1`) |
| Build | ok / failed, image `mc-rag-sim-app`, image size |
| Startup | time to first healthy answer, health status |
| Peak app memory | MB (and when) |
| Peak Qdrant memory | MB |
| CPU | average and peak per container; throttling visible? |
| Throughput | achieved req/s vs offered rate, per run |
| Latency | p50 / p95 / p99 per run |
| Errors | HTTP error rate, 429 count |
| Stability | OOM kills, restarts, stayed up? |
| Bottleneck | the first component to saturate, with the evidence |
| Verdict | fits / fits with a caveat / does not fit — and the next test to run |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose config` fails with `env file .env not found` | `.env` missing | copy `.env.example` to `.env` and fill in the tokens |
| `compose < 2.24 does not support !override` | old Compose | upgrade Docker Desktop |
| Ports not overridden, `8000` also published | missing `!override` | it is in the overlay; re-run stage 1 |
| `docker info` shows 12 cpus after applying limits | WSL did not restart, or Docker Desktop rewrote `.wslconfig` | quit Docker Desktop, `wsl --shutdown`, start it again, re-check |
| Build killed / OOM during `build app` | the 4 GB ceiling applied too early | build first, then apply the ceiling (stage 0) |
| App never healthy | models loading, or a failed fast-fail check | read `docker compose ... logs --tail 40 app`; check `EMBED_MODEL`/`EMBED_DIM`/`RERANK_MODEL` match the shipped models |
| 429s in Profile B | the capacity overlay is not applied | `Start-Simulation.ps1 -Profile B`; the script fails fast if the limit is not 1000 |
| Answers say the passages are all it has, `grounded: false` | S0, expected | switch to S1 only for a small, labelled run |
| Health says `degraded` | no provider key (S0), expected | ignore for S0 measurements |
| Peak memory looks tiny | only the main process counted | use the `docker stats` samples the script writes |
| A stage seems to do nothing | `scripts/simulation/*.ps1` not run with `-ExecutionPolicy Bypass` | add the flag, or `Set-ExecutionPolicy -Scope Process Bypass` |
| Windows harness crashes on `resource` | old checkout | the import is guarded; `uv sync --locked` |

---

## What this does not measure

- A real VPS's disk, network and swap behaviour (the VM has `swap=0` on purpose).
- TLS, a reverse proxy, DNS or any public exposure — the simulation is loopback-only.
- Multi-client behaviour: everything arrives from `127.0.0.1`, so Profile A's 429s are a policy
  result, not a multi-tenant result.
- Any conclusion before a run has actually happened: numbers that were not measured are written as
  "not measured", never estimated.
