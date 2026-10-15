"""The load suite on fakes: stats shape and honest peak-RSS reporting (Req 7, Req 7.4).

The harness runs on Windows while the server runs inside the Linux VM, so its own peak RSS says
nothing about the server. Where the measurement is unavailable it must be ``None`` and render as
``n/a``, never as 0 and never quietly omitted from the report.
"""

import pytest

from eval.e2e.cases import Case
from eval.e2e.report import load_markdown
from eval.e2e.system import load
from tests.e2e_helpers import make_driver


@pytest.fixture
async def driver(tmp_path):
    drv = await make_driver(tmp_path)
    yield drv
    await drv.close()
    await drv.pipe.close()


def _cases() -> list[Case]:
    return [
        Case(
            id="l1", question="When is the Python workshop?", lang="en", role="public", expect="any"
        )
    ]


async def test_load_reports_throughput_and_percentiles(driver):
    results, stats = await load(driver, _cases(), concurrency=2, duration_s=0.2)
    assert stats["requests"] > 0
    assert stats["throughput_rps"] > 0
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert stats["rate_limited"] == 0
    assert stats["error_rate"] == 0
    assert [c.status for c in results[0].checks if c.name == "no_errors"] == ["pass"]


async def test_missing_host_rss_is_none_and_renders_as_na(driver, monkeypatch):
    # Windows has no resource.getrusage: the module guard sets it to None (Req 7.1)
    monkeypatch.setattr("eval.e2e.system.resource", None)
    _, stats = await load(driver, _cases(), concurrency=1, duration_s=0.1)
    assert stats["peak_rss_mb"] is None
    markdown = load_markdown(stats)
    assert "load generator peak RSS: n/a MB" in markdown
    assert "0.0 MB" not in markdown
    assert f"{stats['requests']} requests" in markdown  # the rest of the block still renders


def test_report_renders_a_measured_peak_rss():
    stats = {
        "requests": 10,
        "throughput_rps": 1.5,
        "p50_ms": 10.0,
        "p95_ms": 20.0,
        "p99_ms": 25.0,
        "peak_rss_mb": 123.4,
        "stages": {},
    }
    assert "load generator peak RSS: 123.4 MB" in load_markdown(stats)
