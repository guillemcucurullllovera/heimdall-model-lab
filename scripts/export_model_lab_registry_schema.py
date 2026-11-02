#!/usr/bin/env python3
"""Exports `ModelLabRegistryExport`'s JSON Schema to
`schemas/model_lab_registry/ModelLabRegistryExport.v1.json` (committed).

Kept in its own subdirectory for the same reason `schemas/model2/` is
separate from `schemas/`: `tests/unit/test_schema_freshness.py`'s
no-orphan guard is scoped to MODEL-1's own frozen 11-contract list via a
non-recursive `schemas/*.json` glob.

Run after any change to `contracts/model_lab_registry.py`:
`uv run python scripts/export_model_lab_registry_schema.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.model_lab_registry import ModelLabRegistryExport

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas" / "model_lab_registry"


def main() -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    schema = ModelLabRegistryExport.model_json_schema()
    out = SCHEMAS_DIR / "ModelLabRegistryExport.v1.json"
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(SCHEMAS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
