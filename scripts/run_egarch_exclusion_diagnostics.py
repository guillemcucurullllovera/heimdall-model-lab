#!/usr/bin/env python3
"""Remediation of independent-review finding P1-B: quantifies, for EVERY
(ticker, horizon) vol-ladder cell, how many out-of-sample origins are
dropped from the SHARED evaluation sample (naive/EWMA/GARCH_t/EGARCH_t/
Combo all require every model non-NaN at an origin, in
`vol/evaluate.py::run_vol_ladder` and `compare/vol_model_comparison.py`)
SOLELY because EGARCH_t's own plausibility filter excluded that origin --
not because naive/EWMA/GARCH_t had any problem of their own there.

Does not change production behaviour (the shared-sample filter itself is
untouched) -- this is read-only instrumentation reusing the exact same
production forecast functions, reporting what the filter is currently
doing so it is no longer undisclosed. Writes
`reports/egarch_exclusion_diagnostics.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

HORIZONS = [1, 5, 21, 45]
OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "egarch_exclusion_diagnostics.json"


def main() -> None:
    registry = DatasetSnapshotRegistry.load_from_directory()
    cells = []
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        n = len(r)
        for horizon in HORIZONS:
            train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon, min_train_groups=2)
            oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - horizon})

            naive = naive_trailing_rv_forecast(r, horizon)
            ewma = ewma_forecast(r, horizon, lam=0.94)
            garch = gjr_garch_t_forecast(r, horizon, train_ends).variance_forecast
            egarch = egarch_t_forecast(r, horizon, train_ends).variance_forecast

            non_egarch_valid = {
                i for i in oos_indices
                if not np.isnan(naive[i]) and not np.isnan(ewma[i]) and not np.isnan(garch[i])
            }
            shared_valid = {i for i in non_egarch_valid if not np.isnan(egarch[i])}
            excluded = sorted(non_egarch_valid - shared_valid)
            excluded_years = Counter(dates[i][:4] for i in excluded)

            cells.append({
                "ticker": ticker,
                "horizon": horizon,
                "n_oos_without_egarch_gate": len(non_egarch_valid),
                "n_oos_with_egarch_gate_current_production_value": len(shared_valid),
                "excluded_count": len(excluded),
                "excluded_pct": round(100.0 * len(excluded) / len(non_egarch_valid), 2) if non_egarch_valid else 0.0,
                "excluded_by_calendar_year": dict(sorted(excluded_years.items())),
                "excluded_dates_first_last": (
                    [dates[excluded[0]], dates[excluded[-1]]] if excluded else None
                ),
            })
            print(f"{ticker} h={horizon}: dropped {len(excluded)}/{len(non_egarch_valid)} "
                  f"({cells[-1]['excluded_pct']}%) solely due to EGARCH_t's plausibility filter; "
                  f"years={dict(excluded_years)}")

    out = {
        "purpose": "Per-cell quantification of how many OOS origins are removed from EVERY "
        "model's evaluation (naive/EWMA/GARCH_t/EGARCH_t/Combo) solely because EGARCH_t's own "
        "plausibility filter rejected its own value there. See MODEL2G_RESEARCH_LAYERS_ABC.md "
        "and the independent review's finding P1-B for discussion.",
        "cells": cells,
        "cells_with_zero_exclusions": sum(1 for c in cells if c["excluded_count"] == 0),
        "cells_with_nonzero_exclusions": sum(1 for c in cells if c["excluded_count"] > 0),
        "max_excluded_pct": max(c["excluded_pct"] for c in cells),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(f"{out['cells_with_nonzero_exclusions']}/24 cells lose at least one origin solely to "
          f"EGARCH_t's filter; max {out['max_excluded_pct']}% in one cell.")


if __name__ == "__main__":
    main()
