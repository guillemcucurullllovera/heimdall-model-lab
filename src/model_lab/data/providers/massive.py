"""Massive (Polygon-compatible REST) client — new, isolated implementation.

Base URL `https://api.massive.com`, `apiKey` query param from
`MASSIVE_API_KEY`. Prices are represented as strings (never float) per this
repo's `canon/jcs.py` money convention.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx

from model_lab.data.credentials import require_env_credential
from model_lab.data.providers._http import RateLimiter, SourceUnavailableError, retry_with_backoff

BASE_URL = "https://api.massive.com"


class MassiveError(Exception):
    pass


class MassiveAuthError(MassiveError):
    pass


class MassiveEntitlementError(MassiveError):
    pass


@dataclass(frozen=True)
class MassiveBar:
    t: datetime
    o: str
    h: str
    l: str  # noqa: E741
    c: str
    v: str


@dataclass(frozen=True)
class MassiveBarsResult:
    ticker: str
    bars: list[MassiveBar]
    requested_start: date
    requested_end: date
    adjusted: bool
    raw_payload: bytes


class MassiveClient:
    def __init__(self) -> None:
        self._api_key = require_env_credential("MASSIVE_API_KEY")
        self._limiter = RateLimiter(max_calls=5, window_seconds=60.0)  # conservative free-tier pacing
        self._client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MassiveClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        params = dict(params)
        params["apiKey"] = self._api_key

        def _do() -> httpx.Response:
            self._limiter.wait()
            resp = self._client.get(path, params=params)
            if resp.status_code == 429:
                raise SourceUnavailableError(f"rate limited: {path}")
            if resp.status_code == 401:
                raise MassiveAuthError(f"auth failed (401) for {path}")
            # 403 is deliberately NOT raised here: the body must be
            # inspected to disambiguate a real auth failure from an
            # entitlement/plan-limit response (both use 403 on this API) —
            # see the caller's message-sniffing logic below.
            return resp

        return retry_with_backoff(_do, attempts=3)

    def fetch_daily_bars(self, ticker: str, start: date, end: date, *, adjusted: bool = True) -> MassiveBarsResult:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": "50000"}
        resp = self._get(path, params)
        if resp.status_code != 200:
            body_status = None
            with contextlib.suppress(Exception):
                body_status = resp.json().get("status")
            msg = (resp.text or "")[:300].lower()
            if body_status == "NOT_AUTHORIZED" or any(
                w in msg for w in ("plan", "upgrade", "entitl", "not_authorized", "not authorized")
            ):
                raise MassiveEntitlementError(f"{ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
            raise MassiveError(f"{ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        if payload.get("status") == "NOT_AUTHORIZED":
            raise MassiveEntitlementError(f"{ticker}: entitlement denied: {payload}")
        bars = []
        for r in payload.get("results", []) or []:
            bars.append(
                MassiveBar(
                    t=datetime.fromtimestamp(r["t"] / 1000.0, tz=UTC),
                    o=str(r["o"]),
                    h=str(r["h"]),
                    l=str(r["l"]),
                    c=str(r["c"]),
                    v=str(r["v"]),
                )
            )
        return MassiveBarsResult(
            ticker=ticker,
            bars=bars,
            requested_start=start,
            requested_end=end,
            adjusted=adjusted,
            raw_payload=resp.content,
        )

    def list_option_contracts(self, underlying_ticker: str, *, as_of: date | None = None, limit: int = 10) -> httpx.Response:
        """Reference metadata only (available on free tier). `as_of` pins the
        PIT reference date."""
        params: dict[str, str] = {"underlying_ticker": underlying_ticker, "limit": str(limit)}
        if as_of is not None:
            params["as_of"] = as_of.isoformat()
        return self._get("/v3/reference/options/contracts", params)

    def fetch_splits(self, ticker: str, *, limit: int = 1000) -> SplitsResult:
        """Polygon-compatible `/v3/reference/splits` — corporate-action
        (stock split) reference metadata. Reference endpoints are free-tier
        on Polygon-compatible APIs; confirmed live (not assumed) by the
        caller checking `entitlement_denied` on the result."""
        params: dict[str, str] = {"ticker": ticker, "limit": str(limit)}
        resp = self._get("/v3/reference/splits", params)
        if resp.status_code != 200:
            body_status = None
            with contextlib.suppress(Exception):
                body_status = resp.json().get("status")
            msg = (resp.text or "")[:300].lower()
            if body_status == "NOT_AUTHORIZED" or any(
                w in msg for w in ("plan", "upgrade", "entitl", "not_authorized", "not authorized")
            ):
                return SplitsResult(ticker=ticker, splits=[], raw_payload=resp.content, entitlement_denied=True)
            raise MassiveError(f"{ticker}: HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        if payload.get("status") == "NOT_AUTHORIZED":
            return SplitsResult(ticker=ticker, splits=[], raw_payload=resp.content, entitlement_denied=True)
        splits = []
        for r in payload.get("results", []) or []:
            splits.append(
                SplitEvent(
                    ticker=r.get("ticker", ticker),
                    execution_date=date.fromisoformat(r["execution_date"]),
                    split_from=str(r["split_from"]),
                    split_to=str(r["split_to"]),
                )
            )
        return SplitsResult(ticker=ticker, splits=splits, raw_payload=resp.content, entitlement_denied=False)


@dataclass(frozen=True)
class SplitEvent:
    """Polygon/Massive convention (confirmed empirically against USO's real
    2020-04-29 1-for-8 reverse split, `split_from="8"`, `split_to="1"`):
    `split_from` OLD shares become `split_to` NEW shares as of
    `execution_date`. CAUTION (this is exactly the kind of sign error the
    MODEL-1 adversarial review's whole remediation is about avoiding
    silently): the SHARE-COUNT ratio and the PRICE ratio are INVERSES of
    each other, not the same number — `share_count_ratio_new_per_old` and
    `price_adjustment_factor_pre_split` below are deliberately two distinct,
    clearly-named methods so a caller can never accidentally use the wrong
    one for prices (verified independently: 2.13 [2020-04-28 close] *
    price_adjustment_factor_pre_split() [=8] = 17.04, matching the real
    18.01 post-split open to within an ordinary overnight price move)."""

    ticker: str
    execution_date: date
    split_from: str
    split_to: str

    def share_count_ratio_new_per_old(self) -> str:
        """New shares per old share (e.g. 1-for-8 reverse split: 0.125).
        Applies to VOLUME for observations before execution_date (more
        shares existed pre-split, so historical volume must be DIVIDED by
        this ratio -- equivalently multiplied by its reciprocal -- to
        express it in post-split-equivalent share counts)."""
        from decimal import Decimal

        return str(Decimal(self.split_to) / Decimal(self.split_from))

    def price_adjustment_factor_pre_split(self) -> str:
        """The factor to MULTIPLY pre-split PRICES (open/high/low/close) by
        to express them in post-split-equivalent terms (standard backward
        price adjustment). This is the RECIPROCAL of
        `share_count_ratio_new_per_old` -- for a 1-for-8 reverse split this
        is 8, not 0.125. Never use `share_count_ratio_new_per_old` for
        prices; it is the wrong direction (this exact mistake was caught
        and fixed during P0-4 remediation, see
        `scripts/build_corporate_actions_and_adjusted_uso.py`)."""
        from decimal import Decimal

        return str(Decimal(self.split_from) / Decimal(self.split_to))


@dataclass(frozen=True)
class SplitsResult:
    ticker: str
    splits: list[SplitEvent]
    raw_payload: bytes
    entitlement_denied: bool
