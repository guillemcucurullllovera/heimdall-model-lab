"""AT-10: v1 freeze manifest hash verified; no v2 code path reads a v1
verdict as v2 evidence."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from model_lab.gate_v2.v1_freeze import verify_v1_freeze

pytestmark = pytest.mark.acceptance

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v1_freeze_manifest_verifies_with_zero_violations() -> None:
    result = verify_v1_freeze()
    assert result.n_files_checked == 33
    assert result.n_violations == 0, f"v1 freeze VIOLATION(S) detected: {result.violations}"
    assert result.all_ok


def test_no_v2_module_imports_v1_source() -> None:
    """Structural check: no file under src/model_lab/gate_v2/ imports
    anything from heimdall-v2 (which is where v1's graduation.py/
    dsr_harness.py etc. live) -- the only v1 interaction anywhere in
    MODEL-1 is v1_freeze.py's read-only sha256 recomputation, which reads
    RAW BYTES for hashing, never imports/executes v1 code or treats a v1
    verdict as v2 evidence."""
    gate_v2_dir = REPO_ROOT / "src" / "model_lab" / "gate_v2"
    for path in gate_v2_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("heimdall"), f"{path}: imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("heimdall"), f"{path}: imports from {node.module}"


def test_v1_freeze_module_only_reads_bytes_never_executes_v1_code() -> None:
    """v1_freeze.py's only interaction with heimdall-v2 files is
    `Path.read_bytes()` for hashing -- confirmed by source inspection (no
    `import`, `exec`, `importlib`, or `subprocess` call referencing v1
    paths)."""
    source = (REPO_ROOT / "src" / "model_lab" / "gate_v2" / "v1_freeze.py").read_text(encoding="utf-8")
    for forbidden in ("importlib", "exec(", "eval(", "subprocess"):
        assert forbidden not in source, f"v1_freeze.py must not use {forbidden!r}"


def test_gate_result_never_carries_a_v1_verdict_field() -> None:
    """GateResult (the v2 evidence record) has no field for a 'v1 verdict'
    -- structurally impossible to smuggle a v1 result in as v2 evidence.
    v1 items keep the label 'v1 (legacy statistic)' and are never
    re-scored (per §15 point 8)."""
    from model_lab.gate_v2.contracts import GateResult

    fields = set(GateResult.model_fields.keys())
    assert not any("v1" in f.lower() for f in fields)
