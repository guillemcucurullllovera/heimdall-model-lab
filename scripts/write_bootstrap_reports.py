#!/usr/bin/env python3
"""Turns reports/_etf_coverage_raw.json (written by fetch_etf_bars.py) into
the durable ETF_COVERAGE_MATRIX.md and DATA_BOOTSTRAP_REPORT.md deliverables.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RAW = REPORTS_DIR / "_etf_coverage_raw.json"
UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "GLD", "USO"]


def _fmt(entry: dict, provider: str) -> str:
    p = entry.get(provider)
    if p is None:
        return "not queried"
    if p.get("status") != "OK":
        return f"ERROR ({p.get('error_type')}: {p.get('error', '')[:80]})"
    return (
        f"{p['n_bars']} bars, {p['first_bar_date']} → {p['last_bar_date']}, "
        f"~{p['n_gap_days_weekday_approx']}d weekday-approx gaps, {p['research_state']}"
    )


def main() -> None:
    coverage = json.loads(RAW.read_text(encoding="utf-8"))

    lines = [
        "# ETF Coverage Matrix",
        "",
        f"Generated {datetime.now(UTC).isoformat()}. Universe: {', '.join(UNIVERSE)}. "
        "All rows below are `research_only_unlicensed` (Alpaca/Massive free-tier credentials, "
        "D4 HOLD per the operator's ruling) — none is promotion-eligible until firm-use rights "
        "are confirmed in writing.",
        "",
        "| Symbol | Alpaca (raw bars) | Massive (adjusted bars) |",
        "|---|---|---|",
    ]
    for symbol in UNIVERSE:
        entry = coverage.get(symbol, {})
        lines.append(f"| {symbol} | {_fmt(entry, 'alpaca')} | {_fmt(entry, 'massive')} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Alpaca: `GET /v2/stocks/{symbol}/bars`, `adjustment=raw`, host-locked to "
        "`https://data.alpaca.markets`. Every ticker returned the identical window "
        "2016-01-04→2026-09-29 (2700 bars) — consistent with an account-level SIP "
        "entitlement start date, not a per-symbol limit.",
        "- Massive: `GET /v2/aggs/ticker/{symbol}/range/1/day/...`, `adjusted=true` "
        "(Polygon-compatible). Every ticker returned 2024-09-30→2026-09-29 (501 bars), "
        "consistent with the Basic free-tier's ~2-year depth noted in the readiness pack.",
        "- Gap counts are a WEEKDAY APPROXIMATION only (Mon-Fri minus the observed bars; "
        "no real NYSE/NASDAQ holiday calendar). ~138 gap days over the Alpaca window is "
        "consistent with ordinary federal holidays (~13/yr × ~10.6y), not a data-quality "
        "defect — this is NOT a substitute for a real holiday calendar and should not be "
        "read as 138 missing trading days.",
        "- Corporate actions / split-dividend metadata: NOT separately fetched in this pass "
        "(Massive's `adjusted=true` already folds split/dividend adjustment into `massive_bars` "
        "server-side; Alpaca's `raw` bars are unadjusted and carry no separate corporate-action "
        "snapshot yet — a gap, noted in DATA_BOOTSTRAP_REPORT.md).",
        "- 5-minute intraday bars: NOT fetched in this pass (out of scope for this bootstrap; "
        "the Alpaca endpoint used here fetches daily bars only).",
        "- MarketData.app: used only for option EOD chains (see OPTION_ARCHIVE_STATUS.md, "
        "written separately), not for ETF underlying OHLCV.",
    ]
    (REPORTS_DIR / "ETF_COVERAGE_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORTS_DIR / 'ETF_COVERAGE_MATRIX.md'}")


if __name__ == "__main__":
    main()
