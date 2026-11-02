#!/usr/bin/env python3
"""MODEL-2D: does a 2-state Markov-switching regime read add volatility-
forecasting value beyond the existing GARCH filter, across the six-ETF
universe? Research/shadow only; T-STATE/T-DOWNSTREAM remain BLOCKED."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.regime.evaluate import regime_value_add
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "regime_value_add"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    t_start = time.time()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        t0 = time.time()
        try:
            result = regime_value_add(series)
        except ValueError as e:
            print(f"{ticker}: SKIPPED ({e})")
            summary[ticker] = {"status": "SKIPPED", "reason": str(e)}
            continue
        elapsed = time.time() - t0
        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        summary[ticker] = {
            "status": "OK",
            "n_converged_folds": result.n_converged_folds,
            "n_well_identified_folds": result.n_well_identified_folds,
            "n_folds": result.n_folds,
            "n_oos": result.n_oos,
            "qlike_by_model": result.qlike_by_model,
            "dm_regime_vs_model_p": result.dm_regime_vs_model_p,
            "regime_beats_garch_t": result.qlike_by_model["regime_2state"] < result.qlike_by_model["GARCH_t"],
        }
        print(
            f"{ticker}: OK n_oos={result.n_oos} qlike_regime={result.qlike_by_model['regime_2state']:.4f} "
            f"qlike_garch_t={result.qlike_by_model['GARCH_t']:.4f} ({elapsed:.1f}s)"
        )
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
