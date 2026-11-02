from __future__ import annotations

from datetime import date

import httpx
import pytest

from model_lab.data.providers import fred_alfred as fred_mod
from model_lab.data.providers.fred_alfred import ALFRED_ALL_VINTAGES_START, FredClient


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "require_env_credential", lambda name: "fake-key")


def test_all_vintages_query_params_sent() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"count": 1, "offset": 0, "limit": 100000, "observations": []})

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    client.fetch_all_vintages("INDPRO")
    assert captured["params"]["realtime_start"] == ALFRED_ALL_VINTAGES_START
    assert captured["params"]["realtime_end"] == "9999-12-31"


def test_parses_per_row_vintage_windows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "count": 2,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {"date": "2024-01-01", "value": "100.0", "realtime_start": "2024-02-01", "realtime_end": "2024-02-28"},
                    {"date": "2024-01-01", "value": "101.0", "realtime_start": "2024-03-01", "realtime_end": "9999-12-31"},
                ],
            },
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages("CPIAUCSL")
    assert len(result.observations) == 2
    assert result.observations[0].realtime_start == date(2024, 2, 1)
    assert result.observations[1].realtime_end == date(9999, 12, 31)
    assert result.n_distinct_realtime_starts == 2


def test_missing_value_marker_parsed_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "count": 1,
                "offset": 0,
                "limit": 100000,
                "observations": [{"date": "2024-01-01", "value": ".", "realtime_start": "2024-02-01", "realtime_end": "2024-02-28"}],
            },
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages("SOME_SERIES")
    assert result.observations[0].value is None


def test_degrades_gracefully_on_vintage_cap_exceeded() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400,
                json={
                    "error_code": 400,
                    "error_message": "Bad Request. There are 5119 vintage dates in the specified real-time "
                    "period: 1776-07-04 to 9999-12-31. This exceeds the maximum number of vintage dates "
                    "allowed for this file type (2000).",
                },
            )
        return httpx.Response(
            200,
            json={
                "count": 1,
                "offset": 0,
                "limit": 100000,
                "observations": [{"date": "2024-01-01", "value": "4.5", "realtime_start": "2024-09-30", "realtime_end": "9999-12-31"}],
            },
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages("DGS10")
    assert result.vintage_query_degraded is True
    assert result.degraded_reason is not None
    assert len(result.observations) == 1
    assert calls["n"] == 2  # one failed all-vintages call, one degraded fallback call


def test_pagination_across_multiple_pages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(dict(request.url.params).get("offset", "0"))
        if offset == 0:
            return httpx.Response(
                200,
                json={
                    "count": 3,
                    "offset": 0,
                    "limit": 2,
                    "observations": [
                        {"date": "2024-01-01", "value": "1", "realtime_start": "2024-01-01", "realtime_end": "9999-12-31"},
                        {"date": "2024-01-02", "value": "2", "realtime_start": "2024-01-02", "realtime_end": "9999-12-31"},
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "count": 3,
                "offset": 2,
                "limit": 2,
                "observations": [
                    {"date": "2024-01-03", "value": "3", "realtime_start": "2024-01-03", "realtime_end": "9999-12-31"},
                ],
            },
        )

    client = FredClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=fred_mod.FRED_BASE_URL)
    result = client.fetch_all_vintages("SOME_SERIES")
    assert len(result.observations) == 3
    assert result.truncated is False
