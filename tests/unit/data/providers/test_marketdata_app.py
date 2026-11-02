from __future__ import annotations

from datetime import date

import httpx
import pytest

from model_lab.data.providers import marketdata_app as mda_mod
from model_lab.data.providers.marketdata_app import (
    MarketDataAppClient,
    MarketDataAppEntitlementError,
)


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mda_mod, "require_env_credential", lambda name: "fake-token")


def test_date_param_is_forwarded() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"s": "ok", "optionSymbol": []})

    client = MarketDataAppClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=mda_mod.BASE_URL)
    client.fetch_option_chain("SPY", on_date=date(2024, 6, 3))
    assert captured["params"]["date"] == "2024-06-03"


def test_parallel_column_arrays_parsed_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "s": "ok",
                "optionSymbol": ["SPY240621C00500000", "SPY240621C00510000"],
                "bid": [10.5, 5.2],
                "ask": [10.8, 5.5],
                "updated": [1717000000, 1717000001],
            },
        )

    client = MarketDataAppClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=mda_mod.BASE_URL)
    result = client.fetch_option_chain("SPY")
    assert len(result.rows) == 2
    assert result.rows[0].option_symbol == "SPY240621C00500000"
    assert result.rows[0].bid == "10.5"
    assert result.rows[1].option_symbol == "SPY240621C00510000"
    assert result.rows[1].bid == "5.2"


def test_203_labelled_historical_not_realtime() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(203, json={"s": "ok", "optionSymbol": []})

    client = MarketDataAppClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=mda_mod.BASE_URL)
    result = client.fetch_option_chain("SPY")
    assert result.http_status == 203
    assert "not entitled" in result.entitlement_note
    assert "REALTIME" in result.entitlement_note


def test_402_raises_entitlement_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"s": "error", "errmsg": "Free/Trial users can only access up to 1 year of data."})

    client = MarketDataAppClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=mda_mod.BASE_URL)
    with pytest.raises(MarketDataAppEntitlementError):
        client.fetch_option_chain("SPY", on_date=date(2020, 1, 1))


def test_status_not_ok_returns_empty_rows_not_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"s": "no_data"})

    client = MarketDataAppClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=mda_mod.BASE_URL)
    result = client.fetch_option_chain("USO")
    assert result.rows == []
