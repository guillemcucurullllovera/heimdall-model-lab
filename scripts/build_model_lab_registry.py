#!/usr/bin/env python3
"""Deterministic generation command for the Model Lab multi-model
registry/export -- the Command-Center productization checkpoint beyond
MODEL-2 alone.

Run: `uv run python scripts/build_model_lab_registry.py`
Output: `reports/model_lab_registry/registry_export.json`

Read-only with respect to the Model Lab's own evidence. Writes exactly
one file, under `reports/`, inside this repository. Never touches AMC/
Quantum/Macro/Command-Center code or state (none of which exists in this
worktree) and never writes to any Command Center repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.command_center_export.registry_builder import build_model_lab_registry

OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "model_lab_registry" / "registry_export.json"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = build_model_lab_registry(datetime.now(UTC))
    OUT_PATH.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"git_commit: {registry.git_commit}")
    print(f"n_entries: {len(registry.entries)}")
    print(f"task_hierarchy: {{{', '.join(f'{k}: {len(v)}' for k, v in registry.task_hierarchy.items())}}}")
    print(f"status_buckets: {{{', '.join(f'{k}: {len(v)}' for k, v in registry.status_buckets.items() if v)}}}")
    print(f"research_summary: {registry.research_summary.model_dump_json()}")
    print(f"n_charts: {len(registry.charts)}")
    print(f"export_sha256: {registry.export_sha256}")


if __name__ == "__main__":
    main()
