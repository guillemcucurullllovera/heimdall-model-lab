"""FRED / ALFRED client — new, isolated implementation for MODEL-1's macro
PIT panel. Own rate limiter, own cache (under
./local-data/macro_pit/); never touches
heimdall-command-center's macro files/database/worktree — this is a
completely independent implementation, not an import.

Keyed vintage-aware endpoint: GET /fred/series/observations with
realtime_start=1776-07-04&realtime_end=9999-12-31 returns EVERY historical
vintage row, each carrying its own realtime_start/realtime_end window.

Keyless ALFRED CSV fallback: GET alfredgraph.csv?id=&vintage_date= for a
single point-in-time cross-check when the keyed vintage-depth reading is
inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from model_lab.data.credentials import require_env_credential
from model_lab.data.providers._http import RateLimiter, SourceUnavailableError, retry_with_backoff

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
ALFRED_KEYLESS_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"

ALFRED_ALL_VINTAGES_START = "1776-07-04"
ALFRED_ALL_VINTAGES_END = "9999-12-31"


class FredError(Exception):
    pass


class FredAuthError(FredError):
    pass


class AlfredError(Exception):
    pass


@dataclass(frozen=True)
class VintageObservation:
    series_id: str
    observation_period: date
    value: str | None  # None for FRED's "." missing marker
    realtime_start: date
    realtime_end: date


@dataclass(frozen=True)
class VintageFetchResult:
    series_id: str
    observations: list[VintageObservation]
    raw_payload: bytes
    n_distinct_realtime_starts: int
    earliest_observation_period: date | None
    latest_observation_period: date | None
    earliest_realtime_start: date | None
    latest_realtime_start: date | None
    vintage_query_degraded: bool = False
    degraded_reason: str | None = None
    total_rows_available: int | None = None  # FRED's own `count` field
    truncated: bool = False


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


class FredClient:
    def __init__(self) -> None:
        self._api_key = require_env_credential("FRED_API_KEY")
        self._limiter = RateLimiter(max_calls=20, window_seconds=60.0)
        self._client = httpx.Client(base_url=FRED_BASE_URL, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FredClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, str]) -> httpx.Response:
        params = dict(params)
        params["api_key"] = self._api_key
        params["file_type"] = "json"

        def _do() -> httpx.Response:
            self._limiter.wait()
            resp = self._client.get(path, params=params)
            if resp.status_code == 429:
                raise SourceUnavailableError(f"rate limited: {path}")
            if resp.status_code in (401, 403):
                raise FredAuthError(f"auth failed ({resp.status_code}) for {path}")
            return resp

        return retry_with_backoff(_do, attempts=3)

    def fetch_all_vintages(
        self,
        series_id: str,
        *,
        max_rows: int = 300_000,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> VintageFetchResult:
        """The exact ALFRED PIT pattern: request the full vintage window and
        read each row's own realtime_start/realtime_end (not a single global
        window). FRED caps a single request at 2000 distinct vintage dates —
        a daily market series (yields, spreads) that is effectively
        unrevised can exceed that cap simply from having one "vintage" per
        trading day; when that happens this degrades gracefully to a
        current-vintage-only fetch (honest, not a silent truncation) rather
        than raising, and flags `vintage_query_degraded=True` with the
        reason, so the coverage report can say exactly why full per-vintage
        history wasn't measured for that series.

        `observation_start`/`observation_end` (optional) restrict which
        OBSERVATION PERIODS' vintage histories are requested (NOT which
        vintage/realtime dates — those stay maximal at
        ALFRED_ALL_VINTAGES_START/END). Used by `fetch_all_vintages_full_history`
        to partition a series whose full (observation x vintage) cross-product
        exceeds `max_rows` in one request (e.g. NFCI: 925,555 rows) into
        observation-year chunks, each well under the cap — see
        MODEL1_ADVERSARIAL_REVIEW.md finding D.3/Q.1: a flat row cap on the
        unpartitioned query always drops the MOST RECENT observations first
        (FRED returns rows in observation_period-ascending order), which is
        exactly the defect being fixed here."""
        params: dict[str, str] = {
            "series_id": series_id,
            "realtime_start": ALFRED_ALL_VINTAGES_START,
            "realtime_end": ALFRED_ALL_VINTAGES_END,
        }
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            params["observation_end"] = observation_end.isoformat()
        resp = self._get("/series/observations", params)
        degraded = False
        degraded_reason: str | None = None
        if resp.status_code == 400 and "exceeds the maximum number of vintage dates" in resp.text:
            degraded = True
            degraded_reason = (
                "FRED caps a single vintage-window request at 2000 distinct vintage dates; "
                f"{series_id} exceeded it (consistent with an effectively-unrevised daily "
                "market series where every trading day looks like a new vintage) — degraded "
                "to a current-vintage-only fetch."
            )
            today = date.today()
            resp = self._get(
                "/series/observations",
                {"series_id": series_id, "realtime_start": today.isoformat(), "realtime_end": today.isoformat()},
            )
        if resp.status_code != 200:
            raise FredError(f"{series_id}: HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        total_count: int | None = payload.get("count")

        def _rows_from(p: dict) -> list[VintageObservation]:
            out = []
            for row in p.get("observations", []):
                val = row.get("value")
                out.append(
                    VintageObservation(
                        series_id=series_id,
                        observation_period=_parse_date(row["date"]),
                        value=None if val in (None, ".") else str(val),
                        realtime_start=_parse_date(row["realtime_start"]),
                        realtime_end=_parse_date(row["realtime_end"]),
                    )
                )
            return out

        obs: list[VintageObservation] = _rows_from(payload)
        truncated = False
        page_limit = int(payload.get("limit") or len(obs) or 1)
        offset = page_limit
        # Paginate through FRED's own count/offset/limit envelope when a
        # series' vintage cross-product exceeds one page (observed on NFCI:
        # count=925555 vs a 100000-row page limit). Capped at `max_rows` as
        # a safety net; a series that would exceed the cap is reported
        # truncated=True rather than silently stopping.
        while (
            not degraded
            and total_count is not None
            and offset < total_count
            and len(obs) < max_rows
        ):
            page_params = dict(params)
            page_params["offset"] = str(offset)
            resp = self._get("/series/observations", page_params)
            if resp.status_code != 200:
                raise FredError(f"{series_id}: HTTP {resp.status_code} during pagination: {resp.text[:300]}")
            page = resp.json()
            obs.extend(_rows_from(page))
            offset += int(page.get("limit") or page_limit)
        if total_count is not None and len(obs) < total_count:
            truncated = True

        realtime_starts = {o.realtime_start for o in obs}
        obs_periods = [o.observation_period for o in obs]
        return VintageFetchResult(
            series_id=series_id,
            observations=obs,
            raw_payload=resp.content,
            n_distinct_realtime_starts=len(realtime_starts),
            earliest_observation_period=min(obs_periods) if obs_periods else None,
            latest_observation_period=max(obs_periods) if obs_periods else None,
            earliest_realtime_start=min(realtime_starts) if realtime_starts else None,
            total_rows_available=total_count,
            truncated=truncated,
            latest_realtime_start=max(realtime_starts) if realtime_starts else None,
            vintage_query_degraded=degraded,
            degraded_reason=degraded_reason,
        )

    def fetch_all_vintages_full_history(
        self,
        series_id: str,
        *,
        series_start: date,
        series_end: date,
        chunk_years: int = 5,
        max_rows_per_chunk: int = 300_000,
    ) -> ChunkedVintageFetchResult:
        """Full-history vintage fetch that AVOIDS the single-request row cap
        by partitioning on OBSERVATION PERIOD (not vintage date) into
        `chunk_years`-year windows and concatenating. This is the fix for
        MODEL1_ADVERSARIAL_REVIEW.md finding D.3/Q.1: `fetch_all_vintages`
        alone, capped at `max_rows`, always keeps the EARLIEST observations
        and silently drops the most recent ones (FRED returns rows in
        observation_period-ascending order) — for NFCI (count=925,555) that
        meant the stored PIT panel covered only 1971-1987, with NO current-
        day value. Chunking by observation-year keeps each request's row
        count to that chunk's own revision depth (a small fraction of the
        full cross-product), so both the earliest AND the most recent
        observations survive."""
        all_obs: list[VintageObservation] = []
        raw_payloads: list[bytes] = []
        chunks: list[dict[str, str | int | bool]] = []
        cur_start = series_start
        while cur_start <= series_end:
            cur_end = min(date(cur_start.year + chunk_years - 1, 12, 31), series_end)
            result = self.fetch_all_vintages(
                series_id,
                max_rows=max_rows_per_chunk,
                observation_start=cur_start,
                observation_end=cur_end,
            )
            all_obs.extend(result.observations)
            raw_payloads.append(result.raw_payload)
            chunks.append(
                {
                    "observation_start": cur_start.isoformat(),
                    "observation_end": cur_end.isoformat(),
                    "n_rows": len(result.observations),
                    "total_rows_available": result.total_rows_available or 0,
                    "truncated": result.truncated,
                    "vintage_query_degraded": result.vintage_query_degraded,
                }
            )
            cur_start = date(cur_end.year + 1, 1, 1)

        # Deterministic de-dup: chunks partition on non-overlapping
        # observation-period ranges, so duplicates should not occur across
        # chunks, but de-dup defensively on the natural key anyway (belt and
        # suspenders against an off-by-one chunk boundary).
        seen: set[tuple[date, date, date]] = set()
        deduped: list[VintageObservation] = []
        for o in all_obs:
            key = (o.observation_period, o.realtime_start, o.realtime_end)
            if key not in seen:
                seen.add(key)
                deduped.append(o)

        realtime_starts = {o.realtime_start for o in deduped}
        obs_periods = [o.observation_period for o in deduped]
        any_truncated = any(bool(c["truncated"]) for c in chunks)
        any_degraded = any(bool(c["vintage_query_degraded"]) for c in chunks)
        return ChunkedVintageFetchResult(
            series_id=series_id,
            observations=deduped,
            raw_payloads=raw_payloads,
            chunks=chunks,
            n_distinct_realtime_starts=len(realtime_starts),
            earliest_observation_period=min(obs_periods) if obs_periods else None,
            latest_observation_period=max(obs_periods) if obs_periods else None,
            earliest_realtime_start=min(realtime_starts) if realtime_starts else None,
            latest_realtime_start=max(realtime_starts) if realtime_starts else None,
            any_chunk_truncated=any_truncated,
            any_chunk_degraded=any_degraded,
            total_rows=len(deduped),
        )


@dataclass(frozen=True)
class ChunkedVintageFetchResult:
    series_id: str
    observations: list[VintageObservation]
    raw_payloads: list[bytes]
    chunks: list[dict[str, str | int | bool]]
    n_distinct_realtime_starts: int
    earliest_observation_period: date | None
    latest_observation_period: date | None
    earliest_realtime_start: date | None
    latest_realtime_start: date | None
    any_chunk_truncated: bool
    any_chunk_degraded: bool
    total_rows: int


def fetch_alfred_keyless_csv(series_id: str, vintage_date: date) -> bytes:
    """Keyless ALFRED CSV fallback for a single point-in-time cross-check."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            ALFRED_KEYLESS_URL,
            params={"id": series_id, "vintage_date": vintage_date.isoformat()},
            headers={"User-Agent": "model-lab-pit-macro/0.1"},
        )
        if resp.status_code != 200:
            raise AlfredError(f"{series_id}@{vintage_date}: HTTP {resp.status_code}")
        return resp.content
