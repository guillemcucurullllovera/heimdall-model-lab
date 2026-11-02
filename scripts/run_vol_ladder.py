#!/usr/bin/env python3
"""MODEL-2A mission item C: runs the volatility forecast ladder (naive
trailing RV, EWMA_0.94, GARCH_t, EGARCH_t) walk-forward, for every horizon
T-VOL itself declares ([1, 5, 21, 45] days), across the six-ETF universe
T-VOL/T-TAIL's own `universe.instruments` names. Writes one evidence JSON
per (ticker, horizon) plus a combined summary.

Research/shadow only -- does not touch T-VOL's task status, AMC, or any
promotion path. Bounded, sequential (no parallel host load): ~9s per
(ticker, horizon) pair x 6 tickers x 4 horizons ~= 4 minutes total.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series
from model_lab.vol.evaluate import run_vol_ladder

HORIZONS = [1, 5, 21, 45]
OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "vol_ladder"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    t_start = time.time()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        summary[ticker] = {}
        for horizon in HORIZONS:
            t0 = time.time()
            try:
                result = run_vol_ladder(series, horizon)
            except ValueError as e:
                print(f"{ticker} h={horizon}: SKIPPED ({e})")
                summary[ticker][str(horizon)] = {"status": "SKIPPED", "reason": str(e)}
                continue
            elapsed = time.time() - t0
            out_path = OUT_DIR / f"{ticker}_h{horizon}.json"
            out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
            best_model = min(result.results, key=lambda m: result.results[m]["qlike_mean"])
            summary[ticker][str(horizon)] = {
                "status": "OK",
                "n_oos": result.n_oos_total,
                "oos_start": result.oos_start_date,
                "oos_end": result.oos_end_date,
                "qlike_by_model": {m: r["qlike_mean"] for m, r in result.results.items()},
                "dm_p_vs_naive": {m: r["dm_vs_naive_p"] for m, r in result.results.items()},
                "best_model": best_model,
                "no_leakage_proof_all_passed": (
                    result.no_leakage_proof["naive_prefix_invariance_all_passed"]
                    and result.no_leakage_proof["ewma_prefix_invariance_all_passed"]
                    and result.no_leakage_proof["naive_future_shuffle_invariant"]
                    and result.no_leakage_proof["ewma_future_shuffle_invariant"]
                    and all(c["match"] for c in result.no_leakage_proof["garch_t_truncation_spot_checks"])
                    # Remediation (P1-4): EGARCH_t is now included -- it was
                    # silently absent from this field before, despite being
                    # the single best model on some (ticker, horizon) rows.
                    and all(c["match"] for c in result.no_leakage_proof["egarch_t_truncation_spot_checks"])
                ),
            }
            print(
                f"{ticker} h={horizon}: OK n_oos={result.n_oos_total} best={best_model} "
                f"qlike={result.results[best_model]['qlike_mean']:.4f} ({elapsed:.1f}s)"
            )
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
