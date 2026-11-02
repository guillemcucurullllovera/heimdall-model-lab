#!/usr/bin/env python3
"""Targeted rebuild of NFCI's macro PIT snapshot only (P0-4 remediation
companion / review item Q.1) -- reuses `pit_macro.py`'s real chunked-fetch
code path without re-fetching the other 10 already-healthy series."""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.data import pit_macro

ORIGINAL_FIELDS = dict(pit_macro.MACRO_STATE_VECTOR_FIELDS)


def main() -> None:
    pit_macro.MACRO_STATE_VECTOR_FIELDS.clear()
    pit_macro.MACRO_STATE_VECTOR_FIELDS["financial_conditions"] = "NFCI"
    try:
        snapshots, coverage = pit_macro.build_macro_pit_panel()
    finally:
        pit_macro.MACRO_STATE_VECTOR_FIELDS.clear()
        pit_macro.MACRO_STATE_VECTOR_FIELDS.update(ORIGINAL_FIELDS)

    snap = snapshots[0]
    out_dir = Path("./local-data/snapshots") / snap.snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(snap.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    c = coverage[0]
    print(f"NFCI rebuilt: n_observations={c.n_observations}, earliest={c.earliest_pit_date}, "
          f"latest={c.latest_pit_date}, n_vintages={c.n_vintages}, truncated={c.truncated}, "
          f"total_rows_available={c.total_rows_available}, reason={c.reason!r}")

    # Merge into the existing macro_pit_coverage.json / MACRO_PIT_COVERAGE.md
    # (all 11 fields), replacing only the NFCI row.
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    coverage_json_path = reports_dir / "macro_pit_coverage.json"
    existing = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    existing["fields"] = [f for f in existing["fields"] if f["series"] != "NFCI"] + [c.to_dict()]
    existing["fields"].sort(key=lambda f: list(pit_macro.MACRO_STATE_VECTOR_FIELDS.values()).index(f["series"]) if f["series"] in pit_macro.MACRO_STATE_VECTOR_FIELDS.values() else 999)
    coverage_json_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"updated {coverage_json_path}")


if __name__ == "__main__":
    main()
