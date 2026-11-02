"""Model-Lab-registry analogue of `test_schema_freshness_model2.py`,
scoped to `schemas/model_lab_registry/`."""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.model_lab_registry import ModelLabRegistryExport

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas" / "model_lab_registry"

MODELS = {"ModelLabRegistryExport.v1": ModelLabRegistryExport}


def test_committed_registry_schema_matches_a_fresh_regeneration(tmp_path: Path) -> None:
    mismatches: list[str] = []
    for name, model in MODELS.items():
        fresh = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed_path = SCHEMAS_DIR / f"{name}.json"
        assert committed_path.exists(), f"missing committed schema: {committed_path}"
        if committed_path.read_text(encoding="utf-8") != fresh:
            mismatches.append(name)
            (tmp_path / f"{name}.fresh.json").write_text(fresh, encoding="utf-8")
    assert not mismatches, (
        f"committed registry schema(s) stale: {mismatches} -- run "
        f"`uv run python scripts/export_model_lab_registry_schema.py` and commit the result"
    )


def test_no_orphaned_registry_schema_files() -> None:
    expected = {f"{name}.json" for name in MODELS}
    actual = {p.name for p in SCHEMAS_DIR.glob("*.json")}
    assert not (actual - expected), f"schemas/model_lab_registry/ contains unexpected file(s): {actual - expected}"


def test_model1_schema_freshness_guard_does_not_see_this_schema_either() -> None:
    model1_schemas_dir = REPO_ROOT / "schemas"
    direct_files = {p.name for p in model1_schemas_dir.glob("*.json")}
    assert "ModelLabRegistryExport.v1.json" not in direct_files
