#!/usr/bin/env python3
"""MODEL-2A mission item D: runs the tail-risk ladder (unconditional_GPD_POT,
GJR_t_filtered_GPD_FHS) walk-forward at 95/97.5/99% VaR/ES, across the
six-ETF universe. Writes one evidence JSON per ticker plus a combined
summary. Research/shadow only."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tail.evaluate import run_tail_ladder
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "tail_ladder"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    t_start = time.time()
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        t0 = time.time()
        result = run_tail_ladder(series)
        elapsed = time.time() - t0
        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        summary[ticker] = {
            "n_oos": result.n_oos_total,
            "oos_start": result.oos_start_date,
            "oos_end": result.oos_end_date,
            "kupiec_p_by_model_and_level": {
                m: {p: lvl["kupiec_lr_uc_p"] for p, lvl in levels.items()} for m, levels in result.results.items()
            },
            "cc_p_by_model_and_level": {
                m: {p: lvl["christoffersen_lr_cc_p"] for p, lvl in levels.items()} for m, levels in result.results.items()
            },
            "observed_rate_by_model_and_level": {
                m: {p: lvl["observed_rate"] for p, lvl in levels.items()} for m, levels in result.results.items()
            },
        }
        print(f"{ticker}: OK n_oos={result.n_oos_total} ({elapsed:.1f}s)")
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
