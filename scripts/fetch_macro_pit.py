#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from model_lab.data.pit_macro import build_macro_pit_panel, write_coverage_reports

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
SNAPSHOTS_OUT = Path("./local-data/snapshots")


def main() -> None:
    snapshots, coverage = build_macro_pit_panel()
    for snap in snapshots:
        d = SNAPSHOTS_OUT / snap.snapshot_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_path, md_path = write_coverage_reports(coverage, REPORTS_DIR)
    for c in coverage:
        print(c.field, c.series_id, "n_obs=", c.n_observations, "usable=", c.usable, "reason=", c.reason[:80])
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
