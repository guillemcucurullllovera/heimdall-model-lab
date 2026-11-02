#!/usr/bin/env python3
"""MODEL-2A data unlock (mission item B): the true CBOE VIX index and its
family (VIX3M/VVIX/SKEW) remain RESTRICTED/UNRESOLVED at both providers this
estate has credentials for -- confirmed live in this session (see
reports/VIX_FAMILY_DATA_INVESTIGATION.md): Massive's Polygon-compatible
Indices API lists `I:VIX` in its reference catalog (HTTP 200, read-only
metadata) but returns `403 NOT_AUTHORIZED` on the actual aggregates endpoint
("upgrade your plan"). Genuinely not acquirable today without a plan change
this session is not authorized to make.

What IS genuinely, legitimately available under the SAME entitlement as the
existing six-ETF universe: the four major exchange-traded VIX-family
products (VIXY, UVXY, SVXY, VXX) trade as ordinary equities/ETPs, so
Alpaca's plain `/v2/stocks/{symbol}/bars` endpoint serves them exactly like
SPY/QQQ/etc -- confirmed live. These are NOT the VIX index (they track VIX
futures, carry substantial roll/contango decay, and are legitimately only a
volatility-*regime* proxy, not a real implied-vol level) -- documented
honestly, never relabelled as "implied volatility" data.

These products also undergo FREQUENT reverse splits (UVXY: 13 in the
observed history: SVXY: 5; VIXY: 6; VXX: 8) -- exactly the corporate-action
hazard the P0-4/P1 remediation exists to catch, and a genuine stress test of
the MODEL-2A content-checked companion-dataset fix (contracts/task.py).
Every one of these four tickers gets a real corporate-actions snapshot and a
derived, split-adjusted series -- none is used raw.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from model_lab.contracts.dataset_snapshot import PitRow
from model_lab.contracts.licence import LicenceStatus
from model_lab.data.providers.alpaca import AlpacaClient, AlpacaError
from model_lab.data.providers.massive import MassiveClient
from model_lab.data.snapshot import build_etf_bar_snapshot, build_reference_snapshot, store_raw

UNIVERSE = ["VIXY", "UVXY", "SVXY", "VXX"]
REQUESTED_START = date(1990, 1, 1)
REQUESTED_END = date(2026, 9, 29)
SNAPSHOTS_ROOT = Path("./local-data/snapshots")
TRANSFORM_VERSION = "split_adjust_backward.v1"  # same algorithm/version as build_corporate_actions_and_adjusted_uso.py


def _write_manifest(snap) -> None:
    out_dir = SNAPSHOTS_ROOT / snap.snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _alpaca_bar_to_pit_row(symbol: str, bar, raw_hash: str) -> PitRow:
    return PitRow(
        series_id=symbol,
        observation_period=bar.t.date(),
        value=json.dumps({"o": bar.o, "h": bar.h, "l": bar.l, "c": bar.c, "v": bar.v}),
        realtime_start=None,
        realtime_end=None,
        knowledge_time_utc=bar.t,
        source_snapshot_sha=raw_hash,
    )


def fetch_raw_bars(client: AlpacaClient) -> dict[str, dict]:
    coverage: dict[str, dict] = {}
    for symbol in UNIVERSE:
        try:
            result = client.fetch_daily_bars(symbol, REQUESTED_START, REQUESTED_END, adjustment="raw")
            raw_hashes = [store_raw(p) for p in result.raw_payloads]
            rows = [_alpaca_bar_to_pit_row(symbol, b, raw_hashes[0]) for b in result.bars]
            snap = build_etf_bar_snapshot(
                snapshot_id=f"alpaca_{symbol}_daily_raw",
                source_id="alpaca_bars",
                symbol=symbol,
                endpoint_or_path=f"/v2/stocks/{symbol}/bars",
                params={"start": str(REQUESTED_START), "end": str(REQUESTED_END), "timeframe": "1Day", "adjustment": "raw"},
                licence_status=LicenceStatus.UNRESOLVED,
                rows=rows,
                raw_payload_hashes=raw_hashes,
                requested_start=REQUESTED_START,
                adjustment_kind="raw",
            )
            _write_manifest(snap)
            coverage[symbol] = {
                "status": "OK",
                "n_bars": len(rows),
                "first_bar_date": str(snap.first_bar_date) if snap.first_bar_date else None,
                "last_bar_date": str(snap.last_bar_date) if snap.last_bar_date else None,
                "first_bar_lag_days": snap.first_bar_lag_days,
            }
            print(f"alpaca {symbol}: OK n_bars={len(rows)} first={snap.first_bar_date} last={snap.last_bar_date}")
        except (AlpacaError, Exception) as e:  # noqa: BLE001 - recorded, not raised
            coverage[symbol] = {"status": "ERROR", "error_type": type(e).__name__, "error": str(e)[:400]}
            print(f"alpaca {symbol}: ERROR {type(e).__name__}: {e}")
    return coverage


def fetch_corporate_actions(client: MassiveClient) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for ticker in UNIVERSE:
        r = client.fetch_splits(ticker)
        raw_hash = store_raw(r.raw_payload)
        rows = [
            PitRow(
                series_id=ticker,
                observation_period=s.execution_date,
                value=json.dumps(
                    {
                        "split_from": s.split_from,
                        "split_to": s.split_to,
                        "share_count_ratio_new_per_old": s.share_count_ratio_new_per_old(),
                        "price_adjustment_factor_pre_split": s.price_adjustment_factor_pre_split(),
                    }
                ),
                realtime_start=None,
                realtime_end=None,
                knowledge_time_utc=datetime.combine(s.execution_date, datetime.min.time(), tzinfo=UTC),
                source_snapshot_sha=raw_hash,
            )
            for s in r.splits
        ]
        snap = build_reference_snapshot(
            snapshot_id=f"corporate_actions_{ticker}",
            source_id="massive_reference_splits",
            endpoint_or_path="/v3/reference/splits",
            params={"ticker": ticker},
            licence_status=LicenceStatus.UNRESOLVED,
            rows=rows,
            raw_payload_hashes=[raw_hash],
            schema_id="corporate_action_splits.v1",
        )
        _write_manifest(snap)
        results[ticker] = {"entitlement_denied": r.entitlement_denied, "n_splits": len(r.splits)}
        print(f"{ticker}: corporate_actions snapshot written, {len(r.splits)} split(s) found")
    return results


def build_split_adjusted_series(ticker: str) -> int:
    raw_path = SNAPSHOTS_ROOT / f"alpaca_{ticker}_daily_raw" / "rows.ndjson"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    ca_manifest = json.loads((SNAPSHOTS_ROOT / f"corporate_actions_{ticker}" / "manifest.json").read_text(encoding="utf-8"))
    ca_rows_path = SNAPSHOTS_ROOT / f"corporate_actions_{ticker}" / "rows.ndjson"
    ca_rows = [json.loads(line) for line in ca_rows_path.read_text(encoding="utf-8").splitlines()] if ca_rows_path.exists() else []

    split_events = sorted(
        (
            (
                date.fromisoformat(r["observation_period"]),
                Decimal(json.loads(r["value"])["price_adjustment_factor_pre_split"]),
                Decimal(1) / Decimal(json.loads(r["value"])["price_adjustment_factor_pre_split"]),
            )
            for r in ca_rows
        ),
        key=lambda t: t[0],
    )

    adjusted_rows: list[PitRow] = []
    for raw in raw_rows:
        obs_date = date.fromisoformat(raw["observation_period"])
        bar = json.loads(raw["value"])
        price_factor = Decimal(1)
        share_count_ratio = Decimal(1)
        for exec_date, p_factor, sc_ratio in split_events:
            if obs_date < exec_date:
                price_factor *= p_factor
                share_count_ratio *= sc_ratio
        if price_factor == 1:
            adjusted_bar = bar
        else:
            adjusted_bar = {
                "o": str(Decimal(bar["o"]) * price_factor),
                "h": str(Decimal(bar["h"]) * price_factor),
                "l": str(Decimal(bar["l"]) * price_factor),
                "c": str(Decimal(bar["c"]) * price_factor),
                "v": str(int(Decimal(bar["v"]) * share_count_ratio)),
            }
        adjusted_rows.append(
            PitRow(
                series_id=ticker,
                observation_period=obs_date,
                value=json.dumps(adjusted_bar),
                realtime_start=None,
                realtime_end=None,
                # Same PIT caveat as build_corporate_actions_and_adjusted_uso.py: this
                # is today's-knowledge backward reconstruction, not a genuine historical
                # PIT series for the adjustment itself.
                knowledge_time_utc=raw["knowledge_time_utc"],
                source_snapshot_sha=raw["source_snapshot_sha"],
            )
        )

    snap = build_etf_bar_snapshot(
        snapshot_id=f"{ticker.lower()}_daily_split_adjusted",
        source_id="alpaca_bars_derived",
        symbol=ticker,
        endpoint_or_path=f"derived:alpaca_{ticker}_daily_raw+corporate_actions_{ticker}",
        params={"transform": TRANSFORM_VERSION, "raw_snapshot_id": f"alpaca_{ticker}_daily_raw"},
        licence_status=LicenceStatus.UNRESOLVED,
        rows=adjusted_rows,
        raw_payload_hashes=[],
        requested_start=None,
        adjustment_kind="split_adjusted",
        adjustment_method=f"{TRANSFORM_VERSION}: backward price adjustment using corporate_actions_{ticker} "
        f"(source={ca_manifest['source_id']}); {len(split_events)} split event(s) applied. "
        "NOTE: knowledge_time_utc inherited from raw bars, NOT re-derived for the adjustment's own "
        "knowability -- same documented limitation as uso_daily_split_adjusted.",
        corporate_actions_snapshot_id=f"corporate_actions_{ticker}",
    )
    _write_manifest(snap)
    print(f"{ticker.lower()}_daily_split_adjusted: {len(adjusted_rows)} rows written, {len(split_events)} split(s) applied")
    return len(adjusted_rows)


def discontinuity_check() -> dict[str, list[dict]]:
    findings: dict[str, list[dict]] = {}
    for ticker in UNIVERSE:
        rows_path = SNAPSHOTS_ROOT / f"{ticker.lower()}_daily_split_adjusted" / "rows.ndjson"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda r: r["observation_period"])
        outliers = []
        prev_close = None
        for r in rows:
            bar = json.loads(r["value"])
            close = float(bar["c"])
            if prev_close is not None and prev_close > 0 and close > 0:
                log_ret = math.log(close / prev_close)
                if abs(log_ret) > math.log(2.0):  # >100% single-day move -- these ARE genuinely volatile products
                    outliers.append({"date": r["observation_period"], "prev_close": prev_close, "close": close, "log_return": log_ret})
            prev_close = close
        findings[ticker] = outliers
        print(f"{ticker} (adjusted): {len(outliers)} outlier day(s) > 100% |log return|" + (f" -> {outliers}" if outliers else ""))
    return findings


def main() -> None:
    with AlpacaClient() as alpaca:
        bar_coverage = fetch_raw_bars(alpaca)
    with MassiveClient() as massive:
        splits_coverage = fetch_corporate_actions(massive)
    for ticker in UNIVERSE:
        if bar_coverage.get(ticker, {}).get("status") == "OK":
            build_split_adjusted_series(ticker)
    findings = discontinuity_check()

    out = {"bar_coverage": bar_coverage, "splits_coverage": splits_coverage, "discontinuity_check": findings}
    out_path = Path(__file__).resolve().parents[1] / "reports" / "_vix_etp_universe_raw.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
