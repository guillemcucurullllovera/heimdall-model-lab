"""MarketData.app client — new, isolated implementation.

Base URL `https://api.marketdata.app`, `token` query param from
`MARKETDATA_TOKEN`. Response bodies are PARALLEL COLUMN ARRAYS (one list per
field, all same length, indexed together), status field `s` must be `"ok"`.

Gap this client closes vs. the estate's own adapter: supports `date=` on the
option-chain endpoint for a true historical EOD chain (the estate's adapter
never sends it, so every call there implicitly asks for "now", which this
account's plan can't actually deliver live — see HTTP 203 handling below).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx

from model_lab.data.credentials import require_env_credential
from model_lab.data.providers._http import RateLimiter, SourceUnavailableError, retry_with_backoff

BASE_URL = "https://api.marketdata.app"


class MarketDataAppError(Exception):
    pass


class MarketDataAppAuthError(MarketDataAppError):
    pass


class MarketDataAppEntitlementError(MarketDataAppError):
    pass


@dataclass(frozen=True)
class OptionChainRow:
    option_symbol: str
    bid: str | None
    ask: str | None
    mid: str | None
    last: str | None
    open_interest: int | None
    volume: int | None
    underlying_price: str | None
    iv: str | None
    updated: datetime | None


@dataclass(frozen=True)
class OptionChainResult:
    underlying: str
    requested_date: date | None
    rows: list[OptionChainRow]
    http_status: int
    entitlement_note: str
    raw_payload: bytes


def _col(payload: dict, field: str, i: int) -> object | None:
    values = payload.get(field)
    if not values or i >= len(values):
        return None
    return values[i]


class MarketDataAppClient:
    def __init__(self) -> None:
        self._token = require_env_credential("MARKETDATA_TOKEN")
        self._limiter = RateLimiter(max_calls=20, window_seconds=60.0)
        self._client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MarketDataAppClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        params = dict(params)
        params["token"] = self._token

        def _do() -> httpx.Response:
            self._limiter.wait()
            resp = self._client.get(path, params=params)
            if resp.status_code == 429:
                raise SourceUnavailableError(f"rate limited: {path}")
            if resp.status_code in (401,):
                raise MarketDataAppAuthError(f"auth failed ({resp.status_code}) for {path}")
            return resp

        return retry_with_backoff(_do, attempts=3)

    def fetch_option_chain(
        self, underlying: str, *, on_date: date | None = None, limit: int = 200
    ) -> OptionChainResult:
        """GET /v1/options/chain/{underlying}/?limit=&date= . `date=` (when
        given) requests that past day's 16:00 ET EOD chain — the gap the
        estate's own adapter never closes. Without `date=`, this account
        (Starter Trial: no OPRA/IEX/UTP entitlement, confirmed live) gets a
        stale/cached response labelled HTTP 203 by the vendor; both 200 and
        203 are treated as structurally successful but HISTORICAL, never
        REALTIME/LIVE."""
        params: dict[str, str] = {"limit": str(limit)}
        if on_date is not None:
            params["date"] = on_date.isoformat()
        resp = self._get(f"/v1/options/chain/{underlying}/", params)
        entitlement_note = (
            "HTTP 203: Non-Authoritative Information — this account is not entitled to "
            "OPRA/IEX/UTP; response is stale/cached, never treat as REALTIME"
            if resp.status_code == 203
            else f"HTTP {resp.status_code}"
        )
        if resp.status_code == 402:
            raise MarketDataAppEntitlementError(
                f"{underlying}: HTTP 402 (payment required / plan limit): {resp.text[:300]}"
            )
        if resp.status_code not in (200, 203):
            raise MarketDataAppError(f"{underlying}: HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        if payload.get("s") != "ok":
            return OptionChainResult(
                underlying=underlying,
                requested_date=on_date,
                rows=[],
                http_status=resp.status_code,
                entitlement_note=f"{entitlement_note}; body status={payload.get('s')!r} (no data)",
                raw_payload=resp.content,
            )
        n = len(payload.get("optionSymbol", []) or [])
        rows = []
        for i in range(n):
            updated_raw = _col(payload, "updated", i)
            updated = (
                datetime.fromtimestamp(float(str(updated_raw)), tz=UTC)
                if updated_raw is not None
                else None
            )
            rows.append(
                OptionChainRow(
                    option_symbol=str(_col(payload, "optionSymbol", i)),
                    bid=_stringify(_col(payload, "bid", i)),
                    ask=_stringify(_col(payload, "ask", i)),
                    mid=_stringify(_col(payload, "mid", i)),
                    last=_stringify(_col(payload, "last", i)),
                    open_interest=_intify(_col(payload, "openInterest", i)),
                    volume=_intify(_col(payload, "volume", i)),
                    underlying_price=_stringify(_col(payload, "underlyingPrice", i)),
                    iv=_stringify(_col(payload, "iv", i)),
                    updated=updated,
                )
            )
        return OptionChainResult(
            underlying=underlying,
            requested_date=on_date,
            rows=rows,
            http_status=resp.status_code,
            entitlement_note=entitlement_note,
            raw_payload=resp.content,
        )


def _stringify(v: object | None) -> str | None:
    return None if v is None else str(v)


def _intify(v: object | None) -> int | None:
    return None if v is None else int(float(str(v)))
