"""MODEL-2 analogue of `test_schema_freshness.py`, scoped to
`schemas/model2/` (deliberately separate from `schemas/`'s own MODEL-1-only
freshness/no-orphan guard -- see `scripts/export_model2_export_schema.py`'s
own docstring for why: a MODEL-2 schema committed directly into `schemas/`
previously tripped `test_no_orphaned_committed_schema_files`, confirmed
live before this file was moved into its own subdirectory)."""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.model2_export import Model2CommandCenterExport

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL2_SCHEMAS_DIR = REPO_ROOT / "schemas" / "model2"

MODELS = {"Model2CommandCenterExport.v1": Model2CommandCenterExport}


def test_committed_model2_schema_matches_a_fresh_regeneration(tmp_path: Path) -> None:
    mismatches: list[str] = []
    for name, model in MODELS.items():
        fresh = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed_path = MODEL2_SCHEMAS_DIR / f"{name}.json"
        assert committed_path.exists(), f"missing committed schema: {committed_path}"
        if committed_path.read_text(encoding="utf-8") != fresh:
            mismatches.append(name)
            (tmp_path / f"{name}.fresh.json").write_text(fresh, encoding="utf-8")
    assert not mismatches, (
        f"committed MODEL-2 schema(s) stale: {mismatches} -- run "
        f"`uv run python scripts/export_model2_export_schema.py` and commit the result"
    )


def test_no_orphaned_model2_schema_files() -> None:
    expected = {f"{name}.json" for name in MODELS}
    actual = {p.name for p in MODEL2_SCHEMAS_DIR.glob("*.json")}
    orphaned = actual - expected
    assert not orphaned, f"schemas/model2/ contains file(s) no longer produced: {orphaned}"


def test_model1_schema_freshness_guard_does_not_see_the_model2_schema() -> None:
    """Confirms the separation this subdirectory relies on actually holds
    -- schemas/*.json's own non-recursive glob (used by
    test_schema_freshness.py::test_no_orphaned_committed_schema_files)
    must not pick up anything under schemas/model2/."""
    model1_schemas_dir = REPO_ROOT / "schemas"
    direct_files = {p.name for p in model1_schemas_dir.glob("*.json")}
    assert "Model2CommandCenterExport.v1.json" not in direct_files
