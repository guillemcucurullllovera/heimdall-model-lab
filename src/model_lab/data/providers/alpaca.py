"""Alpaca market-data client — new, isolated implementation for MODEL-1.

Does NOT import from heimdall-v2. Host-locked to `https://data.alpaca.markets`
(the market-data host; the trading hosts `paper-api`/`api.alpaca.markets`
are rejected outright — mirrors the estate's own `_assert_data_host`
fail-closed pattern, independently re-implemented here).

Credentials: `ALPACA_MARKET_DATA_KEY` / `ALPACA_MARKET_DATA_SECRET` (headers
`APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`), `ALPACA_MARKET_DATA_BASE_URL`
(optional; must equal the allowed host or is rejected), `ALPACA_OPTION_FEED`
(optional; only `"indicative"` is accepted — this account's entitlement).
Never logs/prints a credential value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import httpx

from model_lab.data.credentials import read_env_credential, require_env_credential
from model_lab.data.providers._http import RateLimiter, SourceUnavailableError, retry_with_backoff

ALLOWED_HOST = "https://data.alpaca.markets"


class AlpacaError(Exception):
    pass


class AlpacaAuthError(AlpacaError):
    pass


class AlpacaHostViolationError(AlpacaError):
    pass


class AlpacaFeedViolationError(AlpacaError):
    pass


@dataclass(frozen=True)
class AlpacaBar:
    t: datetime  # bar timestamp (session date at midnight UTC for daily bars)
    o: str
    h: str
    l: str  # noqa: E741
    c: str
    v: int


@dataclass(frozen=True)
class AlpacaBarsResult:
    symbol: str
    bars: list[AlpacaBar]
    requested_start: date
    requested_end: date
    adjustment: str
    raw_payloads: list[bytes]  # one per page (Alpaca paginates via next_page_token)


def _assert_data_host(base_url: str) -> str:
    if base_url != ALLOWED_HOST:
        raise AlpacaHostViolationError(
            f"ALPACA_MARKET_DATA_BASE_URL resolved to a non-data host; refusing "
            f"(got a value that is not exactly {ALLOWED_HOST!r})"
        )
    return base_url


def read_credentials() -> tuple[str, str, str]:
    key = require_env_credential("ALPACA_MARKET_DATA_KEY")
    secret = require_env_credential("ALPACA_MARKET_DATA_SECRET")
    base_url = read_env_credential("ALPACA_MARKET_DATA_BASE_URL") or ALLOWED_HOST
    base_url = _assert_data_host(base_url)
    return key, secret, base_url


def read_option_feed() -> str:
    feed = read_env_credential("ALPACA_OPTION_FEED") or "indicative"
    if feed != "indicative":
        raise AlpacaFeedViolationError(
            f"ALPACA_OPTION_FEED must be 'indicative' for this account's entitlement; refusing {feed!r}"
        )
    return feed


class AlpacaClient:
    def __init__(self) -> None:
        key, secret, base_url = read_credentials()
        _assert_data_host(base_url)
        self._base_url = base_url
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._limiter = RateLimiter(max_calls=30, window_seconds=60.0)
        self._client = httpx.Client(base_url=base_url, headers=self._headers, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AlpacaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        _assert_data_host(self._base_url)

        def _do() -> httpx.Response:
            self._limiter.wait()
            resp = self._client.get(path, params=params)
            if resp.status_code == 429:
                raise SourceUnavailableError(f"rate limited: {path}")
            if resp.status_code in (401, 403):
                raise AlpacaAuthError(f"auth failed ({resp.status_code}) for {path}")
            return resp

        return retry_with_backoff(_do, attempts=3)

    def fetch_daily_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        adjustment: str = "raw",
    ) -> AlpacaBarsResult:
        """GET /v2/stocks/{symbol}/bars — new endpoint (not present in the
        estate's own adapter, which only does real-time snapshots). Paginates
        via `next_page_token` until exhausted; `adjustment` in
        {raw, split, dividend, all}."""
        if adjustment not in ("raw", "split", "dividend", "all"):
            raise ValueError(f"invalid adjustment {adjustment!r}")
        bars: list[AlpacaBar] = []
        raw_payloads: list[bytes] = []
        page_token: str | None = None
        while True:
            params = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "timeframe": "1Day",
                "adjustment": adjustment,
                "limit": "10000",
            }
            if page_token:
                params["page_token"] = page_token
            resp = self._get(f"/v2/stocks/{symbol}/bars", params)
            if resp.status_code != 200:
                raise AlpacaError(f"{symbol}: HTTP {resp.status_code} fetching bars: {resp.text[:300]}")
            raw_payloads.append(resp.content)
            payload = resp.json()
            for b in payload.get("bars", []) or []:
                bars.append(
                    AlpacaBar(
                        t=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                        o=str(b["o"]),
                        h=str(b["h"]),
                        l=str(b["l"]),
                        c=str(b["c"]),
                        v=int(b["v"]),
                    )
                )
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return AlpacaBarsResult(
            symbol=symbol,
            bars=bars,
            requested_start=start,
            requested_end=end,
            adjustment=adjustment,
            raw_payloads=raw_payloads,
        )
