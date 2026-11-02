"""P1-6: committed JSON Schema freshness — adversarial review finding D.6
("Committed JSON Schema `schemas/OptionSnapshotRecord.v1.json` is stale
relative to the live code... No test checks committed-schema freshness").

Regenerates every schema into an isolated temp directory (NEVER writes to
the real `schemas/` tree from a test) and diffs byte-for-byte against the
committed files. A stale committed schema fails this test, not just a
manual `git diff` someone has to remember to check.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_json_schema.py"


def _load_export_script_module():
    spec = importlib.util.spec_from_file_location("export_json_schema", EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_committed_schema_matches_a_fresh_regeneration(tmp_path: Path) -> None:
    export_module = _load_export_script_module()
    mismatches: list[str] = []
    missing: list[str] = []

    for name, model in export_module.MODELS.items():
        fresh = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        committed_path = SCHEMAS_DIR / f"{name}.json"
        if not committed_path.exists():
            missing.append(name)
            continue
        committed = committed_path.read_text(encoding="utf-8")
        if committed != fresh:
            mismatches.append(name)
            # Write the fresh version to tmp_path only, for a human to diff
            # locally if this test fails -- never to the real schemas/ tree.
            (tmp_path / f"{name}.fresh.json").write_text(fresh, encoding="utf-8")

    assert not missing, f"schemas/ is missing committed files for: {missing} (run scripts/export_json_schema.py)"
    assert not mismatches, (
        f"committed schema(s) are stale relative to the live pydantic contracts: {mismatches} "
        f"-- run `uv run python scripts/export_json_schema.py` and commit the result. "
        f"(Fresh versions written to {tmp_path} for inspection, not to the source tree.)"
    )


def test_no_orphaned_committed_schema_files() -> None:
    """Every file in schemas/ must correspond to a model still exported by
    the script -- catches a renamed/removed contract leaving a stale file
    behind."""
    export_module = _load_export_script_module()
    expected_files = {f"{name}.json" for name in export_module.MODELS}
    actual_files = {p.name for p in SCHEMAS_DIR.glob("*.json")}
    orphaned = actual_files - expected_files
    assert not orphaned, f"schemas/ contains file(s) no longer produced by export_json_schema.py: {orphaned}"
