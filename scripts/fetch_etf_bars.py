#!/usr/bin/env python3
"""Bootstrap: fetch maximum available daily OHLCV history for the 6-ETF
universe from Alpaca and Massive, build DatasetSnapshots, and print a
coverage summary (also written to reports/ETF_COVERAGE_MATRIX.md by
scripts/write_bootstrap_reports.py, which consumes this script's JSON
output)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from model_lab.contracts.dataset_snapshot import PitRow
from model_lab.contracts.licence import LicenceStatus
from model_lab.data.providers.alpaca import AlpacaClient, AlpacaError
from model_lab.data.providers.massive import MassiveClient, MassiveEntitlementError, MassiveError
from model_lab.data.snapshot import build_etf_bar_snapshot, store_raw

UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "GLD", "USO"]
REQUESTED_START = date(1990, 1, 1)
REQUESTED_END = date(2026, 9, 29)

OUT_DIR = Path("./local-data/snapshots")
COVERAGE_JSON = Path(__file__).resolve().parents[1] / "reports" / "_etf_coverage_raw.json"


def _alpaca_bar_to_pit_row(symbol: str, bar, raw_hash: str) -> PitRow:
    obs_date = bar.t.date()
    return PitRow(
        series_id=symbol,
        observation_period=obs_date,
        value=json.dumps({"o": bar.o, "h": bar.h, "l": bar.l, "c": bar.c, "v": bar.v}),
        realtime_start=None,
        realtime_end=None,
        knowledge_time_utc=bar.t,
        source_snapshot_sha=raw_hash,
    )


def _massive_bar_to_pit_row(symbol: str, bar, raw_hash: str) -> PitRow:
    obs_date = bar.t.date()
    return PitRow(
        series_id=symbol,
        observation_period=obs_date,
        value=json.dumps({"o": bar.o, "h": bar.h, "l": bar.l, "c": bar.c, "v": bar.v}),
        realtime_start=None,
        realtime_end=None,
        knowledge_time_utc=bar.t,
        source_snapshot_sha=raw_hash,
    )


def main() -> None:
    coverage: dict[str, dict] = {}

    with AlpacaClient() as alpaca:
        for symbol in UNIVERSE:
            entry: dict = {"symbol": symbol}
            try:
                result = alpaca.fetch_daily_bars(symbol, REQUESTED_START, REQUESTED_END, adjustment="raw")
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
                (OUT_DIR / snap.snapshot_id).mkdir(parents=True, exist_ok=True)
                (OUT_DIR / snap.snapshot_id / "manifest.json").write_text(
                    json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                entry["alpaca"] = {
                    "status": "OK",
                    "n_bars": len(rows),
                    "first_bar_date": str(snap.first_bar_date) if snap.first_bar_date else None,
                    "last_bar_date": str(snap.last_bar_date) if snap.last_bar_date else None,
                    "first_bar_lag_days": snap.first_bar_lag_days,
                    "n_gap_days_weekday_approx": sum((g.end - g.start).days + 1 for g in snap.known_gaps),
                    "licence_status": "UNRESOLVED",
                    "research_state": "research_only_unlicensed",
                }
            except (AlpacaError, Exception) as e:  # noqa: BLE001 - deliberately broad, recorded not raised
                entry["alpaca"] = {"status": "ERROR", "error_type": type(e).__name__, "error": str(e)[:400]}
            coverage[symbol] = entry
            print(f"alpaca {symbol}: {entry['alpaca'].get('status')} n_bars={entry['alpaca'].get('n_bars')}")

    with MassiveClient() as massive:
        for symbol in UNIVERSE:
            try:
                result = massive.fetch_daily_bars(symbol, REQUESTED_START, REQUESTED_END, adjusted=True)
                raw_hash = store_raw(result.raw_payload)
                rows = [_massive_bar_to_pit_row(symbol, b, raw_hash) for b in result.bars]
                snap = build_etf_bar_snapshot(
                    snapshot_id=f"massive_{symbol}_daily_adjusted",
                    source_id="massive_bars",
                    symbol=symbol,
                    endpoint_or_path=f"/v2/aggs/ticker/{symbol}/range/1/day/{REQUESTED_START}/{REQUESTED_END}",
                    params={"adjusted": "true"},
                    licence_status=LicenceStatus.UNRESOLVED,
                    rows=rows,
                    raw_payload_hashes=[raw_hash],
                    requested_start=REQUESTED_START,
                    adjustment_kind="split_dividend_adjusted",
                )
                (OUT_DIR / snap.snapshot_id).mkdir(parents=True, exist_ok=True)
                (OUT_DIR / snap.snapshot_id / "manifest.json").write_text(
                    json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                coverage[symbol]["massive"] = {
                    "status": "OK",
                    "n_bars": len(rows),
                    "first_bar_date": str(snap.first_bar_date) if snap.first_bar_date else None,
                    "last_bar_date": str(snap.last_bar_date) if snap.last_bar_date else None,
                    "first_bar_lag_days": snap.first_bar_lag_days,
                    "n_gap_days_weekday_approx": sum((g.end - g.start).days + 1 for g in snap.known_gaps),
                    "licence_status": "UNRESOLVED",
                    "research_state": "research_only_unlicensed",
                }
            except (MassiveEntitlementError, MassiveError, Exception) as e:  # noqa: BLE001
                coverage[symbol]["massive"] = {"status": "ERROR", "error_type": type(e).__name__, "error": str(e)[:400]}
            print(f"massive {symbol}: {coverage[symbol]['massive'].get('status')} n_bars={coverage[symbol]['massive'].get('n_bars')}")

    COVERAGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_JSON.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {COVERAGE_JSON}")


if __name__ == "__main__":
    main()
