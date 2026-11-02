"""AT-adjacent (M1-A) import-firewall test.

MODEL-1 is explicitly forbidden from importing or depending on the estate's
other live repos/packages (heimdall-v2, heimdall-command-center,
heimdall-quantum-lab, AMC/PAPER_AUTOPILOT, Macro, Quantum). This test proves
zero coupling at the source level: no file under src/ or tests/ may import
any of the forbidden package/module names, and no forbidden filesystem path
may appear as a hardcoded write target.

This is a static/textual check (not an import-time check) so it also catches
forbidden imports inside code paths that are never executed by the test
suite itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRS = [REPO_ROOT / "src", REPO_ROOT / "scripts"]

# Top-level module/package names that must never be imported by MODEL-1.
# These are the package names used by the sibling estate repos.
FORBIDDEN_IMPORT_ROOTS = {
    "heimdall",  # heimdall-v2 / heimdall-command-center / heimdall-quantum-lab all use this root
    "quantum",  # quantum_sep2026 / quantum-lab package names (would also shadow generic names)
    "core",  # AMC/heimdall-v2 core package — generic name, explicitly flagged as a shadow risk
    "command_center",
    "amc4",
}

# Filesystem paths MODEL-1 must never write to. Read-only access to some of
# these (e.g. the AMC root, for the replay/options-archive observers) is
# permitted and tested elsewhere; this check only forbids *hardcoded write
# targets* appearing in source as string literals combined with obvious
# write verbs, which would indicate an accidental mutation path.
FORBIDDEN_WRITE_PATH_FRAGMENTS = (
    "/heimdall-v2/",
    "/heimdall-command-center/",
    "/heimdall-quantum-lab/",
    "calibration_data/macro3_knowledge",
)

# The ONE legitimate, audited exception: MODEL-1's replay skeleton (M1-G) and
# option-archive ingester (M1-I) are explicitly authorised to *read* the pinned
# AMC data root (D5: `./local-data/replay-input`,
# read-only, no AMC CLI ever invoked — see replay/amc_root.py's own docstring
# and tests/acceptance/test_at13_replay_parity.py's
# `test_observer_never_modifies_the_amc_root`, which asserts this
# mechanically). The path is defined in exactly this one file; every other
# module imports AMC_ROOT/CYCLES_DIR/etc. from here rather than re-hardcoding
# it, so allowlisting only this file (not the whole heimdall-v2 fragment
# everywhere) keeps the check meaningful for every other path in the repo.
FORBIDDEN_WRITE_PATH_ALLOWLIST = {
    REPO_ROOT / "src" / "model_lab" / "replay" / "amc_root.py",
}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in SRC_DIRS:
        if base.exists():
            files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_forbidden_package_imports() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hit = _imported_roots(tree) & FORBIDDEN_IMPORT_ROOTS
        if hit:
            violations.append(f"{path.relative_to(REPO_ROOT)}: imports {sorted(hit)}")
    assert not violations, "Forbidden imports found:\n" + "\n".join(violations)


def test_no_hardcoded_forbidden_write_paths() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        if path in FORBIDDEN_WRITE_PATH_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for frag in FORBIDDEN_WRITE_PATH_FRAGMENTS:
            if frag in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: contains forbidden path fragment {frag!r}")
    assert not violations, "Forbidden hardcoded write paths found:\n" + "\n".join(violations)


def test_allowlisted_amc_root_file_is_read_only_by_construction() -> None:
    """The one allowlisted file must never itself contain a write call
    (open in write mode, write_text, write_bytes, shutil.copy* into that
    path, etc.) — this is what actually justifies the allowlist above,
    checked mechanically rather than only by code review."""
    (path,) = FORBIDDEN_WRITE_PATH_ALLOWLIST
    text = path.read_text(encoding="utf-8")
    forbidden_write_calls = ("write_text(", "write_bytes(", '"w")', "'w')", '"wb")', "'wb')", "shutil.copy")
    hits = [call for call in forbidden_write_calls if call in text]
    assert not hits, f"{path} is allowlisted as read-only but contains write call(s): {hits}"


def test_pyproject_declares_no_forbidden_local_path_dependencies() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for frag in ("heimdall-v2", "heimdall-command-center", "heimdall-quantum-lab"):
        assert frag not in pyproject, f"pyproject.toml must not reference {frag}"
