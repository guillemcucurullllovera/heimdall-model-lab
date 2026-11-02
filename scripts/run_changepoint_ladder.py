#!/usr/bin/env python3
"""MODEL-2C: CUSUM + BOCPD changepoint detection across the six-ETF
universe. Research/shadow only; T-CHANGE remains BLOCKED."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.changepoint.evaluate import run_changepoint_detectors
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "changepoint_ladder"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    t_start = time.time()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        t0 = time.time()
        result = run_changepoint_detectors(series)
        elapsed = time.time() - t0
        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        summary[ticker] = {
            "n_cusum_alarms": result.n_cusum_alarms,
            "n_bocpd_dates_covered": result.n_bocpd_dates_covered,
            "bocpd_top_decile_hits": result.bocpd_top_decile_hits,
            "bocpd_top_5pct_hits": result.bocpd_top_5pct_hits,
            "cusum_alarm_within_5d_of_illustrative": result.cusum_alarm_within_5d_of_illustrative,
        }
        print(
            f"{ticker}: n_cusum_alarms={result.n_cusum_alarms} "
            f"bocpd_top_decile={result.bocpd_top_decile_hits}/{result.n_bocpd_dates_covered} "
            f"cusum_hits={result.cusum_alarm_within_5d_of_illustrative} ({elapsed:.1f}s)"
        )
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
