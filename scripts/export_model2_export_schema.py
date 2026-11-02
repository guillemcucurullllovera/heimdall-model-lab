#!/usr/bin/env python3
"""Exports `Model2CommandCenterExport`'s JSON Schema to
`schemas/model2/Model2CommandCenterExport.v1.json` (committed).

Kept in a SEPARATE subdirectory from `schemas/*.json` deliberately:
`tests/unit/test_schema_freshness.py::test_no_orphaned_committed_schema_files`
asserts that every file DIRECTLY in `schemas/` (non-recursive glob)
corresponds to one of MODEL-1's own frozen, named 11 contracts
(`contracts/__init__.py`'s own docstring) -- adding a MODEL-2-specific
schema there would trip that guard (confirmed live: it did, before this
file was moved into its own subdirectory). `schemas/model2/` is outside
that glob's reach, so this stays a genuinely separate, additive schema
location rather than requiring an edit to that MODEL-1-scoped guard test.

Run after any change to `contracts/model2_export.py`:
`uv run python scripts/export_model2_export_schema.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.model2_export import Model2CommandCenterExport

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas" / "model2"


def main() -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    schema = Model2CommandCenterExport.model_json_schema()
    out = SCHEMAS_DIR / "Model2CommandCenterExport.v1.json"
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(SCHEMAS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
