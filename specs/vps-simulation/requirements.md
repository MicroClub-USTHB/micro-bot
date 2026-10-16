# Requirements Document: 4 GB / 2-core VPS simulation for MC-RAG

## Introduction

MC-RAG must run on a small Linux VPS: 4 GB RAM, 2 CPU cores, everything (app, Qdrant, OS) inside
that budget. We cannot buy the VPS yet, so we need a reproducible way to answer three questions on
the development machine (Windows + Docker Desktop): does the stack fit at all, how fast does it
answer, and which component saturates first.

The approach is a **simulation layer that reuses the production image and services unchanged**:
Compose overlay files that add resource limits, a loopback-only port and a separate environment
file, a separate Compose project so no container, network or volume is shared with the normal
deployment, and the existing `eval/e2e` harness driving load against the real application over
HTTP from Windows. The load generator stays outside the constrained VM; the server's real CPU and
memory are read with `docker stats`.

Nothing here changes how MC-RAG behaves in production. Every artifact is additive: two Compose
overlays, one env template, PowerShell helpers, one runbook, and small changes to the eval harness
(request pacing, honest host-RSS reporting) that are useful outside the simulation too.

## Requirements

### Requirement 1: Isolated simulation stack

**User Story:** As a developer, I want the simulation to be a separate Compose project, so that
starting, benchmarking or destroying it can never touch my normal development deployment.

#### Acceptance Criteria

1.1 WHEN the simulation is started THEN it SHALL use the Compose project name `mc-rag-sim`.
1.2 WHEN the simulation is started THEN its containers, network and named volumes SHALL be prefixed
`mc-rag-sim_`, so `app_data` and `qdrant_storage` resolve to simulation-local storage and never to
the default project's volumes.
1.3 WHEN the simulation runs THEN Qdrant SHALL NOT publish any host port; it SHALL be reachable only
on the simulation's internal Compose network at `http://qdrant:6333`.
1.4 WHEN the simulation runs THEN the app SHALL publish exactly one port, `127.0.0.1:8001` mapped to
container port `8000`, bound to loopback only.
1.5 WHEN the simulation Compose files are merged with `docker-compose.yml` THEN the inherited
`ports` and `env_file` entries SHALL be **replaced**, not appended (Compose `!override`), so the app
never publishes `8000:8000` and never reads the production `.env`.
1.6 WHEN the simulation is stopped or removed with the provided command THEN the default project's
containers and volumes SHALL be untouched.

### Requirement 2: VM-level resource ceiling

**User Story:** As a developer, I want the whole Linux VM limited to 4 GB and 2 CPUs, so that the
measurement includes the operating system and the page cache, not only the containers.

#### Acceptance Criteria

2.1 WHEN the WSL 2 backend is in use THEN the runbook SHALL document `%UserProfile%\.wslconfig`
with `memory=4GB`, `processors=2`, `swap=0`, applied by a helper that backs up, merges, verifies and
restores the previous file, and never overwrites it silently.
2.2 WHEN the helper is asked to apply limits THEN it SHALL print a timestamped backup path and SHALL
refuse to run when no backup was taken in the same session.
2.3 WHEN Docker Desktop does not use the WSL 2 backend THEN the runbook SHALL document the
Hyper-V/`DockerVMSwitch` alternative (`Set-VMMemory`, `Set-VMProcessor`) and SHALL say that the
helper does not apply it.
2.4 WHEN the environment is limited THEN the runbook SHALL explain how to verify the effective
limits (`docker info`, `docker stats`, `/proc/meminfo` inside a container) instead of assuming the
change took effect.
2.5 WHEN the image build or model download is about to run under a 4 GB ceiling THEN the runbook
SHALL instruct building first with the ceiling lifted, then re-applying it before the benchmark.

### Requirement 3: Container budgets inside the VM

**User Story:** As a developer, I want explicit per-container memory and CPU budgets, so that a
runaway component is capped and the bottleneck is attributable.

#### Acceptance Criteria

3.1 WHEN the simulation starts THEN the app SHALL be limited to 2800 MB memory and 1.5 CPUs.
3.2 WHEN the simulation starts THEN Qdrant SHALL be limited to 512 MB memory and 0.5 CPUs.
3.3 WHEN the simulation starts THEN each container's memory limit SHALL be enforced without extra
swap allowance (`memswap_limit` equal to `mem_limit`), so a container is OOM-killed by the runtime
rather than rescued by swap.
3.4 WHEN the documentation is written THEN it SHALL state that these limits sum to ~3.3 GB and that
the remaining ~0.7 GB is the VM's operating system, Docker and page cache, and that the numbers are
an initial hypothesis to be adjusted from measured usage, not a proven fit.
3.5 WHEN the simulation starts THEN Uvicorn SHALL keep its single worker unless a run proves
otherwise.

### Requirement 4: Production secrets and data isolation

**User Story:** As a developer, I want the simulated app to run without any production credential, so
that benchmarking can never spend real quota or leak a key.

#### Acceptance Criteria

4.1 WHEN the simulation runs THEN it SHALL read the existing `.env`, which is already ignored by Git
and by `.dockerignore`; no second environment file SHALL be introduced.
4.2 WHEN the simulation runs THEN every simulation-only value (collection name, thread counts, ingest
concurrency, Profile B's rate limit) SHALL come from the simulation Compose overlay and never from
`.env`, so the normal dev stack keeps its own values.
4.3 WHEN scenario S0 is selected THEN `OPENROUTER_API_KEY` SHALL be absent from the container
environment (commented out in `.env`), and no fake key SHALL be supplied to make a health check look
healthy.
4.4 WHEN the simulation runs THEN `QDRANT_URL=http://qdrant:6333`, a simulation-specific
`QDRANT_COLLECTION` from the overlay and `DATA_DIR`-backed document storage on the simulation volume
SHALL be used.
4.5 WHEN the simulation runs THEN the real embedder (`bge-m3`), the real reranker (mmarco) and the
real BM25 stopwords SHALL be used, with thread counts pinned explicitly
(`INGEST_EMBED_THREADS=2`, `RETRIEVAL_RERANK_THREADS=2`, `RETRIEVAL_RERANK_WORKERS=1`) so ONNX does
not spawn more threads than the VM can serve.
4.6 WHEN the simulation runs THEN local-only `MEMBER_TOKEN` and `ADMIN_API_KEY` values SHALL be used,
and the same values SHALL be supplied to the Windows harness through the environment.
4.7 WHEN the simulation runs THEN authentication, error envelopes and the response contract SHALL
behave exactly as in production.

### Requirement 5: Expected degraded health

**User Story:** As a developer, I want the simulation's health expectations written down, so that a
`degraded` status is not mistaken for a broken retrieval stack.

#### Acceptance Criteria

5.1 WHEN `OPENROUTER_API_KEY` is absent from the environment (S0) THEN `GET /api/v1/health` SHALL
report `degraded` while document listing and retrieval still work, and the runbook SHALL say this is
expected.
5.2 WHEN a chat request is made in the simulation THEN the app SHALL answer through the extractive
fallback with `grounded=false`, zero LLM calls and the passages as sources, and the runbook SHALL
state that this is expected and that timings therefore measure retrieval, not generation.
5.3 WHEN the runbook describes the optional generation scenarios THEN it SHALL require explicit
opt-in for each, SHALL keep them out of the default benchmark, and SHALL require a dedicated test
credential for the external one.

### Requirement 5b: Generation scenarios (none, external provider)

**User Story:** As a developer, I want to benchmark the retrieval stack alone first, then optionally
add generation from the external free provider, so that provider latency and quota never contaminate
the capacity measurement.

#### Acceptance Criteria

5b.1 The runbook SHALL define two scenarios and label which one a given stage uses: **S0** no provider
(`OPENROUTER_API_KEY` commented out in `.env`, extractive fallback, 0 LLM calls) and **S1** external
OpenRouter generation with the free `google/gemma-4-31b-it:free` model and the operator's own key.
Both SHALL be switched by editing one line in `.env` and restarting the app, since settings are read
once at startup.
5b.2 WHEN scenario S0 is used THEN retrieval latency, throughput and error rate SHALL be reported as
the retrieval-stack result, with a statement that generation is excluded by design.
5b.3 WHEN scenario S1 is used THEN it SHALL be run with a bounded case set or low offered rate, SHALL
be documented as consuming the free-tier daily budget (`LLM_DAILY_LIMIT=50`, `LLM_RESERVE=5`, at most
2 calls per question), and SHALL be reported separately from S0 numbers.
5b.4 WHEN scenario S1 is used THEN the report SHALL name the provider latency as an external,
out-of-VM factor that a self-hosted VPS could not have had in the same form.
5b.5 No self-hosted model SHALL be added to the simulation: the generation endpoint, the client and
the settings SHALL stay exactly as they are (no local-llm service, no base-URL setting).


### Requirement 5c: Credential hygiene for the scenarios

**User Story:** As the repository owner, I want no credential to reach git, an image, a log or a shell
history, so that benchmarking never leaks a key.

#### Acceptance Criteria

5c.1 `.env` SHALL remain in `.gitignore` and in `.dockerignore` (it already is), so the key is never
committed and never reaches a build layer through `COPY . /app`.
5c.2 No real key SHALL appear in any tracked file, and no template file carrying a key SHALL be added.
5c.3 WHEN any script prints the resolved Compose configuration or environment THEN it SHALL redact
`OPENROUTER_API_KEY`, `MEMBER_TOKEN`, `ADMIN_API_KEY` and any judge key values, because
`docker compose config` prints them in clear text.
5c.4 The runbook SHALL tell the operator to rotate any key that was pasted into a chat, a ticket or a
terminal, and SHALL NOT echo the key back in any script output.
5c.5 The final diff SHALL be inspected for accidental secret exposure before anything is committed, and
an automated test SHALL scan tracked files for the OpenRouter key prefix.


### Requirement 6: Configurable request pacing in the eval harness

**User Story:** As a developer benchmarking capacity, I want to choose the offered request rate from
the command line, so I can sweep load instead of being pinned to 20 requests per minute.

#### Acceptance Criteria

6.1 WHEN `python -m eval.e2e.run` runs without a pacing argument THEN the HTTP driver SHALL keep
pacing at the current default of 20 requests per 60 seconds.
6.2 WHEN `--requests-per-minute N` is passed with `N > 0` on an HTTP run THEN `HttpDriver` SHALL be
constructed with that offered rate.
6.3 WHEN `--requests-per-minute` is 0 or negative THEN the CLI SHALL fail with a clear error instead
of dividing by zero.
6.4 WHEN several load tasks pace concurrently THEN the driver SHALL keep the offered rate a global
maximum (no burst above it, no lost slots) with an asyncio lock around the pacing decision.
6.5 WHEN the runbook explains a rate THEN it SHALL state that it is a maximum offered rate: achieved
throughput is bounded by latency and saturation, not by the flag alone.

### Requirement 7: Honest host-RSS reporting on Windows

**User Story:** As a developer, I want the load report to be honest about what it could not measure,
so that a missing number is never read as a good one.

#### Acceptance Criteria

7.1 WHEN `eval.e2e.system` runs on Windows THEN it SHALL NOT import-crash on the missing `resource`
module (already guarded; covered by a test).
7.2 WHEN peak host RSS is unavailable THEN `peak_rss_mb` SHALL be `null`, never `0` and never a
fabricated value.
7.3 WHEN the markdown report renders load stats with `peak_rss_mb = null` THEN it SHALL print `n/a`
and the runbook SHALL state that this is the Windows load generator's own memory, not the server's.
7.4 WHEN the benchmark needs the server's memory THEN the runbook SHALL require `docker stats` on
the simulation containers.

### Requirement 8: Two distinct rate-limit profiles

**User Story:** As a developer, I want to separate "how the app behaves under its own policy" from
"what the hardware can do", so 429s are not mistaken for saturation.

#### Acceptance Criteria

8.1 Profile A SHALL run the load suite with the application's configured default rate limit
(20 requests / 60 s) and SHALL report how many responses were 429.
8.2 The runbook SHALL explain that concurrent virtual users share one client IP (127.0.0.1) from the
host, so Profile A cannot model independent remote clients.
8.3 Profile B SHALL use a simulation-only Compose overlay that raises
`CHAT_RATE_LIMIT_REQUESTS=1000` for the 60 s window, and SHALL NOT modify the production default or
`.env.example`.
8.4 The runbook SHALL label Profile B explicitly as a test-only setting that permits resource
measurement and is not a production-recommended security configuration.

### Requirement 9: Reproducible PowerShell runbook

**User Story:** As a developer on Windows, I want exact commands and helper scripts for every stage,
so the benchmark is repeatable without improvising.

#### Acceptance Criteria

9.1 Helpers SHALL cover preflight, build, start, wait-for-health, smoke test, staged benchmark,
`docker stats` sampling, report and cleanup.
9.2 Cleanup SHALL remove only the `mc-rag-sim` project's containers, network and volumes, SHALL print
a warning that `-v` deletes simulation data, and SHALL never target the default project.
9.3 Every helper SHALL be safe to run twice and SHALL fail loudly (non-zero exit) rather than
silently continuing.
9.4 The runbook SHALL list the stages in order with the exact command for each, the expected output,
and how to interpret it.

### Requirement 10: Staged benchmark plan and report

**User Story:** As a developer, I want a staged load ramp with recorded resource usage, so that I
learn the capacity limit instead of guessing it.

#### Acceptance Criteria

10.1 The stages SHALL be: preflight, build, start + initialize, retrieval smoke test, Profile A
(default rate limit), Profile B capacity ramp (1×30 s, 2×60 s, 4×120 s, 8×120 s), ingestion test,
report, cleanup.
10.2 Each capacity run SHALL be labelled and run alone, and the runbook SHALL instruct stopping
escalation on repeated crashes, OOM kills or unresponsiveness.
10.3 The final report SHALL include: limits actually applied, startup outcome, peak app and Qdrant
memory, observed CPU and throttling where measurable, throughput and p50/p95/p99 per run, error
rate, 429 count, restarts, OOM events, whether the service stayed up, and the identified bottleneck
with a recommended next test.
10.4 The report SHALL state that "containers started" is not evidence of success.
10.5 The ingestion stage SHALL be separate from the chat load stage, SHALL use the existing corpus
and ingestion tooling where possible, and SHALL stay inside the simulation instance.

### Requirement 11: Automated verification

**User Story:** As a developer and reviewer, I want the simulation configuration and harness changes
covered by automated checks, so a future change cannot silently break isolation.

#### Acceptance Criteria

11.1 Tests SHALL verify that the default offered rate is still 20 per 60 s and that a custom rate is
applied.
11.2 Tests SHALL verify that a non-positive `--requests-per-minute` is rejected.
11.3 Tests SHALL verify that concurrent pacing does not exceed the offered rate.
11.4 Tests SHALL verify that `eval.e2e.system` imports and reports `null` peak RSS when `resource` is
unavailable, and that the markdown report renders `n/a`.
11.5 Tests SHALL verify the simulation Compose files: project name, loopback-only `127.0.0.1:8001`
port, `!override` of `ports`/`env_file`, memory/CPU/swap limits on both services, no host port for
Qdrant, and that the capacity overlay changes nothing but the chat rate limit.
11.6 Tests SHALL verify that `.env` is listed in `.gitignore` and in `.dockerignore`, and that the
simulation introduces no second environment file.
11.7 The fast suite, ruff check and ruff format check SHALL pass without model files, and
`docker compose -f docker-compose.yml -f docker-compose.simulation.yml config` SHALL validate when
Docker is available.

### Requirement 12: Non-goals and guardrails

**User Story:** As the repo owner, I want the production deployment untouched, so that the simulation
is a measurement tool and not a fork.

#### Acceptance Criteria

12.1 The `Dockerfile`, `docker-compose.yml`, `.env.example` and application source SHALL NOT be
changed for the simulation, except for the eval-harness changes in Requirements 6 and 7.
12.2 No production credential, real `.env` value or benchmark data SHALL be committed, printed or
pushed.
12.3 The simulation SHALL NOT be published on a public interface, and Qdrant SHALL NOT be exposed to
the host.
12.4 No commit, push or pull request SHALL be created without an explicit request.

## Out of scope

- Buying or provisioning a real VPS, DNS, TLS or a public deployment.
- Changing models, dependencies, chunking or the retrieval pipeline.
- Self-hosting a generation model (no local LLM service, no configurable LLM base URL, no
  `config.py` / `llm/client.py` change).
- Load generator redesign: the existing HTTP driver and load suite are reused; only pacing is added.
- A separate benchmarking framework, results database or CI workflow for the simulation.
- Tuning the app for the simulation (thread counts and limits only, both configurable).
