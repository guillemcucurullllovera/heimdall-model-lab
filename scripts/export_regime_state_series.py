#!/usr/bin/env python3
"""Exports a bounded, real regime-state-probability series per ticker, for
the Model Lab registry's regime-model chart (Command Center productization
checkpoint). No committed artifact anywhere in this repository currently
stores the day-by-day `high_vol_prob` series -- `reports/regime_value_add/
*.json` only stores fold-level summaries (checked directly before writing
this script). This calls the EXACT SAME, already-tested
`regime.markov_switching.fit_and_filter_2state` function MODEL-2D/2G
already use -- no new modeling, no new statistic, a bounded recent window
(last 180 trading days) of its own real output made available as a
committed artifact for the first time.

Run: `uv run python scripts/export_regime_state_series.py`
Output: `reports/regime_value_add/{ticker}_recent_state_series.json`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.regime.markov_switching import fit_and_filter_2state
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "regime_value_add"
RECENT_WINDOW = 180


def main() -> None:
    registry = DatasetSnapshotRegistry.load_from_directory()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        dates = series.dates[1:]
        train_ends, _ = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        regime_res = fit_and_filter_2state(series.log_returns, train_ends)

        valid_idx = [i for i in range(len(dates)) if not np.isnan(regime_res.high_vol_prob[i])]
        recent_idx = valid_idx[-RECENT_WINDOW:] if len(valid_idx) > RECENT_WINDOW else valid_idx

        payload = {
            "ticker": ticker,
            "dataset_id": series.dataset_id,
            "data_fingerprint": series.data_fingerprint,
            "window_description": f"last {len(recent_idx)} trading days with a well-identified fold (of {len(valid_idx)} total valid days)",
            "dates": [dates[i] for i in recent_idx],
            "high_vol_prob": [float(regime_res.high_vol_prob[i]) for i in recent_idx],
            "regime_variance_forecast": [float(regime_res.regime_variance_forecast[i]) for i in recent_idx],
        }
        out_path = OUT_DIR / f"{ticker}_recent_state_series.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{ticker}: wrote {len(recent_idx)} days -> {out_path}")


if __name__ == "__main__":
    main()
