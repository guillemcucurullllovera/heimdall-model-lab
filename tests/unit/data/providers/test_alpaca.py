from __future__ import annotations

from datetime import date

import httpx
import pytest

from model_lab.data.providers import alpaca as alpaca_mod
from model_lab.data.providers.alpaca import (
    AlpacaAuthError,
    AlpacaClient,
    AlpacaFeedViolationError,
    AlpacaHostViolationError,
    _assert_data_host,
    read_option_feed,
)


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_mod, "read_credentials", lambda: ("fake-key", "fake-secret", alpaca_mod.ALLOWED_HOST))


def test_host_lock_rejects_trading_host() -> None:
    with pytest.raises(AlpacaHostViolationError):
        _assert_data_host("https://api.alpaca.markets")


def test_host_lock_accepts_only_data_host() -> None:
    assert _assert_data_host("https://data.alpaca.markets") == "https://data.alpaca.markets"


def test_option_feed_rejects_non_indicative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_OPTION_FEED", "opra")
    with pytest.raises(AlpacaFeedViolationError):
        read_option_feed()


def test_option_feed_accepts_indicative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_OPTION_FEED", raising=False)
    assert read_option_feed() == "indicative"


def _bars_response(bars: list[dict], next_page_token: str | None = None) -> httpx.Response:
    body = {"bars": bars, "next_page_token": next_page_token}
    return httpx.Response(200, json=body)


def test_fetch_daily_bars_parses_and_stringifies_prices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _bars_response(
            [{"t": "2024-01-02T05:00:00Z", "o": 470.1, "h": 471.0, "l": 469.5, "c": 470.8, "v": 1000}]
        )

    client = AlpacaClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=alpaca_mod.ALLOWED_HOST)
    result = client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result.bars) == 1
    bar = result.bars[0]
    assert bar.o == "470.1"
    assert isinstance(bar.o, str)


def test_fetch_daily_bars_paginates_via_next_page_token() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        if "page_token" not in request.url.params:
            return _bars_response([{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}], next_page_token="tok2")
        return _bars_response([{"t": "2024-01-03T05:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 2}])

    client = AlpacaClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=alpaca_mod.ALLOWED_HOST)
    result = client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(result.bars) == 2
    assert len(calls) == 2
    assert len(result.raw_payloads) == 2


def test_401_raises_auth_error_not_retried() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(401, json={"message": "unauthorized"})

    client = AlpacaClient()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=alpaca_mod.ALLOWED_HOST)
    with pytest.raises(AlpacaAuthError):
        client.fetch_daily_bars("SPY", date(2024, 1, 1), date(2024, 1, 5))
    assert call_count["n"] == 1  # auth errors are not retried
