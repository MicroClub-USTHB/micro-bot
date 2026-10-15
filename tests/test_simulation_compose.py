"""The simulation Compose overlays: isolation, budgets, test-only overrides (specs/vps-simulation).

These checks need neither Docker nor the models, so they run in the fast suite. They are the guard
that keeps the 4 GB / 2-core simulation from ever reusing the normal deployment's port, volumes,
environment or rate limit.
"""

import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SIMULATION = REPO / "docker-compose.simulation.yml"
CAPACITY = REPO / "docker-compose.simulation.capacity.yml"
BASE_ENV = REPO / ".env.example"

# The overlay's environment block: simulation-only, so the dev stack in .env keeps its own values
SIM_ENV = {
    "QDRANT_COLLECTION": "mcclub_sim",
    "INGEST_EMBED_THREADS": "2",
    "RETRIEVAL_RERANK_THREADS": "2",
    "RETRIEVAL_RERANK_WORKERS": "1",
    "INGEST_MAX_CONCURRENCY": "1",
}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also accepts Compose's ``!override`` / ``!reset`` merge tags."""


def _tagged(loader, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_Loader.add_constructor("!override", _tagged)
_Loader.add_constructor("!reset", lambda loader, node: None)


def _load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


def _env_values(path: Path) -> dict[str, str]:
    """The uncommented ``KEY=VALUE`` pairs of a dotenv file."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _tracked_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / name for name in listed]


def test_simulation_overlay_is_its_own_project_on_loopback_only() -> None:
    config = _load(SIMULATION)
    app = config["services"]["app"]
    assert config["name"] == "mc-rag-sim"
    # !override replaces the inherited "8000:8000"; adding a port would have published both
    assert app["ports"] == ["127.0.0.1:8001:8000"]
    assert app["env_file"] == [".env"]
    assert "build" not in app and "image" not in app  # the production build, reused as is


def test_simulation_overlay_limits_both_services_without_extra_swap() -> None:
    services = _load(SIMULATION)["services"]
    for name, mem, cpus in (("app", 2800, "1.5"), ("qdrant", 512, "0.5")):
        service = services[name]
        assert service["mem_limit"] == f"{mem}m"
        assert service["memswap_limit"] == service["mem_limit"], "swap could mask an OOM"
        assert service["cpus"] == cpus
        assert service["mem_reservation"] <= service["mem_limit"]
    assert "ports" not in services["qdrant"], "Qdrant must stay on the internal network"


def test_simulation_overlay_sets_only_simulation_env() -> None:
    assert _load(SIMULATION)["services"]["app"]["environment"] == SIM_ENV


def test_capacity_overlay_only_raises_the_chat_rate_limit() -> None:
    overlay = _load(CAPACITY)
    assert overlay == {
        "services": {
            "app": {
                "environment": {
                    "CHAT_RATE_LIMIT_REQUESTS": "1000",
                    "CHAT_RATE_LIMIT_WINDOW_S": "60",
                }
            }
        }
    }


def test_simulation_only_env_stays_out_of_the_shared_env_files() -> None:
    # The dev stack reads .env, so a simulation-only value there would change it silently. The
    # example documents the dev values, which must stay different from the overlay's.
    example = _env_values(BASE_ENV)
    assert example["INGEST_EMBED_THREADS"] == "4"
    assert example["RETRIEVAL_RERANK_THREADS"] == "4"
    assert example["QDRANT_COLLECTION"] == "mcclub"
    if (REPO / ".env").exists():  # local and git-ignored: the file the dev stack actually reads
        dev = _env_values(REPO / ".env")
        for key, value in SIM_ENV.items():
            if key in dev:
                assert dev[key] != value, f"{key} in .env would also change the dev stack"


def test_env_stays_ignored_by_git_and_docker() -> None:
    for name in (".gitignore", ".dockerignore"):
        lines = (REPO / name).read_text(encoding="utf-8").splitlines()
        assert ".env" in [line.strip() for line in lines], name


def test_no_second_simulation_env_file() -> None:
    assert [p.name for p in REPO.glob(".env.simulation*")] == []


def test_no_tracked_file_carries_an_openrouter_key() -> None:
    offenders = [p.name for p in _tracked_files() if b"sk-or-v1" in p.read_bytes()]
    assert offenders == []
