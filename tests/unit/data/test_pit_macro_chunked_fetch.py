"""Tests for P0-4/Q.1 remediation: NFCI's chunked full-history vintage
fetch (MODEL1_ADVERSARIAL_REVIEW.md finding D.3 — the flat 300k-row cap
always dropped the most recent observations; the fix partitions by
observation-period year so nothing recent is silently lost).
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from model_lab.data.providers import fred_alfred as fred_mod
from model_lab.data.providers.fred_alfred import FredClient


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "require_env_credential", lambda name: "fake-key")


def _obs(period: str, value: str, rt_start: str, rt_end: str = "9999-12-31") -> dict:
    return {"date": period, "value": value, "realtime_start": rt_start, "realtime_end": rt_end}


def test_chunked_fetch_partitions_by_observation_year_and_concatenates() -> None:
    """A 3-chunk series (chunk_years=1, 2020-2022): each chunk returns one
    observation. Confirms observation_start/observation_end are sent per
    chunk and results are concatenated, not overwritten."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        year = int(params["observation_start"][:4])
        return httpx.Response(
            200,
            json={
                "count": 1,
                "offset": 0,
                "limit": 100000,
                "observations": [_obs(f"{year}-06-15", f"{year}.0", f"{year}-06-16")],
            },
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages_full_history(
        "NFCI", series_start=date(2020, 1, 1), series_end=date(2022, 12, 31), chunk_years=1
    )

    assert len(calls) == 3
    assert [c["observation_start"] for c in calls] == ["2020-01-01", "2021-01-01", "2022-01-01"]
    assert result.total_rows == 3
    assert result.earliest_observation_period == date(2020, 6, 15)
    assert result.latest_observation_period == date(2022, 6, 15)
    assert not result.any_chunk_truncated
    assert not result.any_chunk_degraded


def test_chunked_fetch_reaches_the_final_chunk_unlike_a_flat_row_cap() -> None:
    """Regression for the exact defect: a flat per-request row cap (as the
    old single-request `fetch_all_vintages(max_rows=...)` had) would starve
    later chunks of budget; the chunked fetch applies the cap PER CHUNK, so
    a large early chunk never prevents a later (more recent) chunk from
    being fetched."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        year = int(params["observation_start"][:4])
        # Simulate a much larger first chunk and a tiny final (most recent) chunk.
        n = 20 if year == 2020 else 1
        obs = [_obs(f"{year}-01-{i + 1:02d}", "1.0", f"{year}-01-{i + 1:02d}") for i in range(n)]
        return httpx.Response(200, json={"count": n, "offset": 0, "limit": 100000, "observations": obs})

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages_full_history(
        "NFCI", series_start=date(2020, 1, 1), series_end=date(2021, 12, 31), chunk_years=1, max_rows_per_chunk=300_000
    )
    # Both the large early chunk AND the small later (more recent) chunk survive.
    assert result.latest_observation_period == date(2021, 1, 1)
    assert result.total_rows == 21


def test_chunked_fetch_dedups_defensively_on_boundary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"count": 1, "offset": 0, "limit": 100000, "observations": [_obs("2020-06-15", "1.0", "2020-06-16")]},
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    # Same series+period+vintage returned by two overlapping calls (simulated
    # by a handler that ignores the chunk params) must not be double-counted.
    result = client.fetch_all_vintages_full_history(
        "NFCI", series_start=date(2020, 1, 1), series_end=date(2021, 12, 31), chunk_years=1
    )
    assert result.total_rows == 1


@pytest.mark.network
def test_live_nfci_full_history_reaches_current_vintage() -> None:
    """Live, network-calling confirmation that the real stored NFCI data
    (rebuilt by scripts/rebuild_nfci.py) genuinely reaches a current
    observation -- the specific gap MODEL1_ADVERSARIAL_REVIEW.md finding
    D.3 found. Opt-in only (`pytest -m network`); not part of the default
    `-m "not network"` CI run."""
    with FredClient() as client:
        result = client.fetch_all_vintages_full_history(
            "NFCI", series_start=date(2026, 1, 1), series_end=date.today(), chunk_years=1
        )
    assert result.total_rows > 0
    assert result.latest_observation_period is not None
    assert result.latest_observation_period.year == 2026
    assert not result.any_chunk_truncated
