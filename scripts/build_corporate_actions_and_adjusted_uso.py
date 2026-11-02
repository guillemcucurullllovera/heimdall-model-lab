#!/usr/bin/env python3
"""Remediation for MODEL1_ADVERSARIAL_REVIEW.md finding C.4 (P0-4):
USO's raw Alpaca daily bars contain the real 2020-04-29 1-for-8 reverse
split UNADJUSTED (close 2020-04-28=2.13, open 2020-04-29=18.01, a mechanical
+213% single-day "return"). This script:

1. Fetches real corporate-actions (splits) reference data for all 6 ETFs
   from Massive's Polygon-compatible `/v3/reference/splits` endpoint
   (confirmed live/entitled on this account) and stores it as its own
   DatasetSnapshot per ticker (`corporate_actions_<TICKER>`).
2. Builds a DERIVED, split-adjusted USO series (`uso_daily_split_adjusted`)
   from the immutable `alpaca_USO_daily_raw` rows — the raw snapshot is
   never modified in place; this is a new, separate DatasetSnapshot whose
   `adjustments` block names the raw snapshot + corporate-actions snapshot
   + transformation version it was derived from.
3. Re-checks all 6 tickers for any other single-day discontinuity (>50%
   |log return|) as an independent sanity check, not just USO.
"""

from __future__ import annotations

import json
import math
from datetime import date
from decimal import Decimal
from pathlib import Path

from model_lab.contracts.dataset_snapshot import PitRow
from model_lab.contracts.licence import LicenceStatus
from model_lab.data.providers.massive import MassiveClient
from model_lab.data.snapshot import build_etf_bar_snapshot, build_reference_snapshot, store_raw

UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "GLD", "USO"]
SNAPSHOTS_ROOT = Path("./local-data/snapshots")
TRANSFORM_VERSION = "split_adjust_backward.v1"  # bump this if the adjustment algorithm ever changes


def _write_manifest(snap) -> None:
    out_dir = SNAPSHOTS_ROOT / snap.snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_and_store_corporate_actions(client: MassiveClient) -> dict[str, dict]:
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
                knowledge_time_utc=__import__("datetime").datetime.combine(
                    s.execution_date, __import__("datetime").time(0, 0), tzinfo=__import__("datetime").UTC
                ),
                source_snapshot_sha=raw_hash,
            )
            for s in r.splits
        ]
        snap = build_reference_snapshot(
            snapshot_id=f"corporate_actions_{ticker}",
            source_id="massive_reference_splits",
            endpoint_or_path="/v3/reference/splits",
            params={"ticker": ticker},
            licence_status=LicenceStatus.UNRESOLVED,  # same Massive account terms as the bars
            rows=rows,
            raw_payload_hashes=[raw_hash],
            schema_id="corporate_action_splits.v1",
        )
        _write_manifest(snap)
        results[ticker] = {
            "entitlement_denied": r.entitlement_denied,
            "n_splits": len(r.splits),
            "splits": [
                {
                    "execution_date": str(s.execution_date),
                    "split_from": s.split_from,
                    "split_to": s.split_to,
                    "price_adjustment_factor_pre_split": s.price_adjustment_factor_pre_split(),
                }
                for s in r.splits
            ],
        }
        print(f"{ticker}: corporate_actions snapshot written, {len(r.splits)} split(s) found")
    return results


def build_split_adjusted_uso(splits: list[dict]) -> None:
    raw_path = SNAPSHOTS_ROOT / "alpaca_USO_daily_raw" / "rows.ndjson"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    # Sort split events ascending by execution_date; apply each to every
    # observation STRICTLY BEFORE that execution_date (standard backward
    # price adjustment). Multiple splits compose multiplicatively.
    # price_factor: multiply pre-split PRICES by this (e.g. 8 for USO's
    # 1-for-8 reverse split -- prices go UP when adjusted backward, since
    # post-split each remaining share is worth ~8x as much).
    # share_count_ratio: multiply pre-split VOLUME by this (e.g. 0.125 --
    # volume goes DOWN, expressed in post-split-equivalent share counts).
    # These are RECIPROCALS of each other -- see SplitEvent's docstring in
    # providers/massive.py for why using the wrong one is a real, easy
    # mistake (caught and fixed during this remediation).
    split_events = sorted(
        (
            (
                date.fromisoformat(s["execution_date"]),
                Decimal(s["price_adjustment_factor_pre_split"]),
                Decimal(1) / Decimal(s["price_adjustment_factor_pre_split"]),
            )
            for s in splits
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
                series_id="USO",
                observation_period=obs_date,
                value=json.dumps(adjusted_bar),
                realtime_start=None,
                realtime_end=None,
                # PIT NOTE (documented limitation, not silently glossed): this
                # reuses the ORIGINAL bar's knowledge_time_utc. The adjustment
                # factor for pre-split observations was only actually knowable
                # starting at the split's execution_date (2020-04-29) -- a
                # strict PIT consumer querying, say, 2019 data "as it was known
                # in 2019" would see UNADJUSTED values in reality. This derived
                # snapshot is today's-knowledge backward reconstruction for
                # cross-sectional/tail-modelling research use, not a genuine
                # historical PIT series for the adjustment itself. Flagged
                # explicitly in the snapshot's adjustments.method string and in
                # ETF_COVERAGE_MATRIX.md; no caller should assume otherwise.
                knowledge_time_utc=raw["knowledge_time_utc"],
                source_snapshot_sha=raw["source_snapshot_sha"],
            )
        )

    corp_actions_raw_path = SNAPSHOTS_ROOT / "corporate_actions_USO" / "manifest.json"
    corp_actions_manifest = json.loads(corp_actions_raw_path.read_text(encoding="utf-8"))

    snap = build_etf_bar_snapshot(
        snapshot_id="uso_daily_split_adjusted",
        source_id="alpaca_bars_derived",
        symbol="USO",
        endpoint_or_path="derived:alpaca_USO_daily_raw+corporate_actions_USO",
        params={"transform": TRANSFORM_VERSION, "raw_snapshot_id": "alpaca_USO_daily_raw"},
        licence_status=LicenceStatus.UNRESOLVED,  # derived from Alpaca raw; inherits the same (not-stronger) authority
        rows=adjusted_rows,
        raw_payload_hashes=[],  # no new raw vendor payload -- this is a pure local transformation
        requested_start=None,
        adjustment_kind="split_adjusted",
        adjustment_method=f"{TRANSFORM_VERSION}: backward price adjustment using corporate_actions_USO "
        f"(source={corp_actions_manifest['source_id']}); NOTE: knowledge_time_utc inherited from raw bars, "
        "NOT re-derived for the adjustment's own knowability -- see script docstring PIT caveat",
        corporate_actions_snapshot_id="corporate_actions_USO",
    )
    _write_manifest(snap)
    print(f"uso_daily_split_adjusted: {len(adjusted_rows)} rows written")


def discontinuity_check() -> dict[str, list[dict]]:
    """Independent sanity re-check across all 6 tickers (not just USO) for
    any single-day |log return| > 50%, using each ticker's BEST available
    series (the split-adjusted one for USO, raw Alpaca for the other 5,
    which the review already found clean but this re-verifies rather than
    assuming)."""
    findings: dict[str, list[dict]] = {}
    for ticker in UNIVERSE:
        snap_id = "uso_daily_split_adjusted" if ticker == "USO" else f"alpaca_{ticker}_daily_raw"
        rows_path = SNAPSHOTS_ROOT / snap_id / "rows.ndjson"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda r: r["observation_period"])
        outliers = []
        prev_close = None
        for r in rows:
            bar = json.loads(r["value"])
            close = float(bar["c"])
            if prev_close is not None and prev_close > 0 and close > 0:
                log_ret = math.log(close / prev_close)
                if abs(log_ret) > math.log(1.5):  # >50% single-day move
                    outliers.append({"date": r["observation_period"], "prev_close": prev_close, "close": close, "log_return": log_ret})
            prev_close = close
        findings[ticker] = outliers
        print(f"{ticker} ({snap_id}): {len(outliers)} outlier day(s) > 50% |log return|" + (f" -> {outliers}" if outliers else ""))
    return findings


def main() -> None:
    with MassiveClient() as client:
        splits_by_ticker = fetch_and_store_corporate_actions(client)

    build_split_adjusted_uso(splits_by_ticker["USO"]["splits"])
    findings = discontinuity_check()

    out = {
        "splits_by_ticker": splits_by_ticker,
        "discontinuity_check": findings,
    }
    out_path = Path(__file__).resolve().parents[1] / "reports" / "_corporate_actions_remediation_raw.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
