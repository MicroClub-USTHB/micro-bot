# Implementation Plan: 4 GB / 2-core VPS simulation for MC-RAG

- [x] 1. Make the secret paths safe before the simulation exists
  - Confirm `.env` is listed in `.gitignore` and in `.dockerignore` (the Dockerfile does `COPY . /app`, so an unignored env file would land in a build layer)
  - Add `pyyaml~=6.0` to the `dev` dependency group in `pyproject.toml` and re-run `uv lock`
  - Create `tests/test_simulation_compose.py` with a `!override`-aware `yaml.SafeLoader` subclass and the first two tests: both ignore files list `.env`, no `.env.simulation*` file exists in the repo, and no tracked file (`git ls-files`) contains `sk-or-v1`
  - Run the new test, then make it pass
  - _Requirements: 4.1, 5c.1, 5c.2, 5c.5, 11.5, 11.6_

- [x] 2. Write the simulation Compose overlays
  - Create `docker-compose.simulation.yml`: top-level `name: mc-rag-sim`; `app` with `ports: !override ["127.0.0.1:8001:8000"]`, `env_file: !override [.env]`, `environment:` with exactly `QDRANT_COLLECTION: mcclub_sim`, `INGEST_EMBED_THREADS: "2"`, `RETRIEVAL_RERANK_THREADS: "2"`, `RETRIEVAL_RERANK_WORKERS: "1"`, `INGEST_MAX_CONCURRENCY: "1"`, plus `mem_limit: 2800m`, `mem_reservation: 1024m`, `memswap_limit: 2800m`, `cpus: "1.5"`, `pids_limit: 512`; `qdrant` with `mem_limit: 512m`, `mem_reservation: 256m`, `memswap_limit: 512m`, `cpus: "0.5"`, `pids_limit: 256`; no `image:`/`build:` keys
  - Create `docker-compose.simulation.capacity.yml`: only `services.app.environment` with `CHAT_RATE_LIMIT_REQUESTS: "1000"` and `CHAT_RATE_LIMIT_WINDOW_S: "60"`
  - Extend `tests/test_simulation_compose.py`: project name, single loopback port, `env_file == [".env"]`, `mem_limit == memswap_limit` and `cpus` on both services, Qdrant has no `ports` key, the five simulation-only env keys and their values, that none of them is set in `.env.example`, and that the capacity overlay has exactly the two rate keys
  - _Requirements: 1.1-1.5, 3.1-3.3, 4.2, 4.4, 8.3, 11.5_

- [x] 3. Make the eval harness's offered request rate configurable and concurrency-safe
  - Add `--requests-per-minute` to `eval/e2e/run.py` with a positive-int argparse type, default `20`, passed to `HttpDriver(args.base_url, requests_per_window=..., window_s=60)` on the `http` target only, and recorded as `offered_rpm` in the run `info`
  - Add `self._pace = asyncio.Lock()` to `HttpDriver` and hold it in `_wait()` while reserving the next slot, releasing before the request; document in the class docstring that the rate is a maximum offered rate
  - Extend `tests/test_e2e_driver.py`: default 20/60 gives `interval == 3.0`, a custom rate is applied, `--requests-per-minute 0` and `-5` exit with a usage error, and 8 concurrent `chat()` calls are spaced by at least `interval / 2`
  - _Requirements: 6.1-6.5, 11.1, 11.2, 11.3_

- [x] 4. Report the load generator's peak RSS honestly
  - Add a peak-RSS line to `load_markdown()` in `eval/e2e/report.py`, printed as `n/a` when `stats["peak_rss_mb"]` is `None`, and say in the wording that it is the load generator's own memory
  - Create `tests/test_e2e_system.py`: monkeypatch `eval.e2e.system.resource` to `None` and assert `load()` returns `peak_rss_mb is None` with the other stats intact; assert `load_markdown()` renders `n/a` and still renders throughput and percentiles
  - _Requirements: 7.1-7.4, 11.4_

- [x] 5. Add the VM-limit helper for the WSL 2 backend
  - Create `scripts/simulation/Set-SimVmLimits.ps1` with `-Mode Show|Apply|Restore`: `Show` prints the current file (or "absent"), its `[wsl2]` keys and the effective `docker info` `NCPU`/`MemTotal`; `Apply` backs up `%UserProfile%\.wslconfig` to `.wslconfig.sim-backup-<UTC>` (recording that none existed), merges `memory=4GB`, `processors=2`, `swap=0` into the existing `[wsl2]` section while keeping other keys, and prints the `wsl --shutdown` + Docker Desktop restart steps without executing them; `Restore` puts the backup back or deletes the file, then prints the same steps
  - Make the script refuse to write when the backup step failed, detect a non-WSL-2 backend and point at the Hyper-V `Set-VMMemory` / `Set-VMProcessor` manual path instead
  - _Requirements: 2.1-2.4, 2.3_

- [x] 6. Add the PowerShell stage helpers
  - Create `scripts/simulation/Test-SimPreflight.ps1`: check the CLI, the running daemon, Compose >= 2.24, the presence of `.env`, validate `docker compose config` for base+simulation and base+simulation+capacity, print the resolved limits with every secret value redacted, fail when a simulation-only key is also set in `.env`, and list the default project's containers to show they are untouched; distinct exit codes 2/3/4
  - Create `scripts/simulation/Start-Simulation.ps1`: optional `-Build`, `up -d`, poll `/api/v1/health` up to 300 s, then print `docker compose ps`, the per-container limits and restart counts from `docker inspect`, and the health JSON (redacted)
  - Create `scripts/simulation/Invoke-SimBenchmark.ps1`: parameters `-Scenario S0|S1`, `-Profile A|B`, `-Concurrency`, `-Duration`, `-RequestsPerMinute`, `-Label`; read `MEMBER_TOKEN` and `ADMIN_API_KEY` from `.env` in memory into the child environment, start a `docker stats` sampler writing `eval/results/simulation/<UTC>-<label>.csv`, run `python -m eval.e2e.run --target http --base-url http://127.0.0.1:8001 --suite load ...`, then record OOM kills and restart counts
  - Create `scripts/simulation/Stop-Simulation.ps1`: interactive confirm unless `-Force`, hard-refuse any project name other than `mc-rag-sim`, `down -v`, print what was removed and warn that `-v` deletes simulation data
  - _Requirements: 9.1-9.3, 10.1, 10.2, 10.3_

- [x] 7. Write the runbook
  - Create `docs/simulation/4gb-vps.md` with the baseline numbers measured on this machine (12 CPUs / 7.72 GB without `.wslconfig`), the file list, and the eight stages with the exact PowerShell command, expected output and how to read it: preflight, build, start and initialize (S0 expected `degraded`), retrieval smoke test, Profile A, Profile B ramp (1x30 s, 2x60 s, 4x120 s, 8x120 s), ingestion test, report and cleanup
  - Document S0 and S1 as labelled scenarios, the quota budget for S1, the build-before-limits order, the loopback-only and Qdrant-not-exposed guarantees, the Hyper-V alternative, and the troubleshooting table from the design
  - Document the report template: limits applied, startup outcome, peak app/Qdrant memory, CPU and throttling, throughput and p50/p95/p99 per run, error rate, 429s, restarts, OOM events, uptime, bottleneck, next test — and that "containers started" is not evidence
  - _Requirements: 2.4, 5.1, 5.2, 5.3, 5b.1-5b.4, 8.1, 8.2, 8.4, 9.4, 10.1, 10.3, 10.4, 10.5_

- [x] 8. Keep the repo docs in sync
  - Add EDGE_CASES.md rows with evidence for: Compose merging `ports`/`env_file` without `!override`; the Docker Desktop VM defaulting to 12 CPUs / 7.72 GB with no `.wslconfig`; the unpaced burst race in `HttpDriver._wait()`; `docker compose config` printing every secret in clear text
  - Add the simulation commands to the AGENTS.md Commands section and a dated PROGRESS.md entry
  - Leave README.md, `.env.example`, `Dockerfile`, `docker-compose.yml` and all application code untouched
  - _Requirements: 12.1, 12.2_

- [x] 9. Verify the whole change set
  - Run `uv lock --check`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest -m "not ocr and not tokenizer and not embedder and not reranker"`
  - Validate `docker compose -f docker-compose.yml -f docker-compose.simulation.yml config` and the same command with the capacity overlay, and record the resolved port, env file, limits, network and volume names
  - Review the full diff for changes to production files, secrets and destructive commands, and confirm no task marked a runtime result that was not executed
  - _Requirements: 11.7, 12.1, 12.2, 12.3_
