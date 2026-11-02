#!/usr/bin/env python3
"""MODEL-2E: Hansen SPA + Model Confidence Set across the volatility
ladder's 5 models (naive/EWMA/GARCH_t/EGARCH_t/Combo), for all six ETFs
across all 4 horizons.

Extended from a SPY-only comparison to the full six-ETF universe as Layer-A
research follow-up (this was flagged in the original MODEL-2E report as
"mechanically straightforward future work" -- it is; the only change needed
was looping over `SIX_ETFS` instead of a hardcoded "SPY"). Running the
comparison on every ticker (not just SPY) avoids the obvious critique that a
single-ticker comparison could be an accident of that one series."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.compare.vol_model_comparison import compare_vol_models
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

HORIZONS = [1, 5, 21, 45]
OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "vol_model_comparison"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    t0 = time.time()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        summary[ticker] = {}
        for h in HORIZONS:
            result = compare_vol_models(series, h)
            out_path = OUT_DIR / f"{ticker}_h{h}.json"
            out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
            summary[ticker][str(h)] = {
                "spa_p_value_consistent": result.spa_p_value_consistent,
                "spa_benchmark_not_rejected": result.spa_benchmark_not_rejected,
                "mcs_superior_set": result.mcs_superior_set,
            }
            print(f"{ticker} h={h}: spa_p={result.spa_p_value_consistent:.4f} mcs={result.mcs_superior_set}")
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
