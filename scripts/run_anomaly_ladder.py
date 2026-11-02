#!/usr/bin/env python3
"""MODEL-2B: covariance/dependence anomaly scoring across the six-ETF panel.
Research/shadow only; T-ANOM (C4) remains BLOCKED (see module docstrings)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.anomaly.evaluate import run_anomaly_ladder
from model_lab.anomaly.panel import build_return_panel
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import SIX_ETFS

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "anomaly_ladder"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    t0 = time.time()
    panel = build_return_panel(SIX_ETFS, registry)
    result = run_anomaly_ladder(panel)
    out_path = OUT_DIR / "six_etf_panel.json"
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    print(f"n_folds={result.n_folds} n_oos={result.n_oos_total} elapsed={time.time()-t0:.1f}s")
    for model_id, r in result.results.items():
        print(f"  {model_id}: illustrative_covered={r['illustrative_dates_covered']} top_decile={r['illustrative_dates_in_top_decile']} top_5pct={r['illustrative_dates_in_top_5pct']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
