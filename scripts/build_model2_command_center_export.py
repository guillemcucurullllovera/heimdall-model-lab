#!/usr/bin/env python3
"""Deterministic generation command for the MODEL-2 Command-Center export.

Run: `uv run python scripts/build_model2_command_center_export.py`
Output: `reports/command_center_export/model2_export.json`

Read-only with respect to the Model Lab's own evidence; writes exactly one
file, under `reports/`, inside this repository. Never touches AMC/Quantum/
Macro/Command-Center code or state (none of which exists in this
worktree) and never writes to any Command Center repository -- consuming
this file from a SEPARATE Command Center repository is the intended next
step, explicitly out of this repository's own scope.

`export_sha256` is a content-integrity hash of the FULL document, including
`identity.generated_at_utc` -- it is expected to differ on every run (that is
correct tamper-evidence behaviour, the same convention every other self-
referential hash in this project follows), not a determinism guarantee.
What IS deterministic, re-running with unchanged underlying evidence: every
OTHER field (all 11 sections' actual content) -- each is a pure projection
of already-committed, content-addressed evidence, never recomputed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.command_center_export.builder import build_model2_export

OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "command_center_export" / "model2_export.json"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export = build_model2_export(datetime.now(UTC))
    OUT_PATH.write_text(export.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"schema: {export.schema_name}/{export.schema_version}")
    print(f"model_id: {export.identity.model_id}")
    print(f"lifecycle_state: {export.identity.lifecycle_state}")
    print(f"git_commit: {export.identity.git_commit}")
    print(f"eligible_cells: {len(export.candidate_definition.eligible_cells)}")
    print(f"historical_cells: {len(export.historical_evidence.cells)} "
          f"(better={export.historical_evidence.n_cells_significant_better} "
          f"worse={export.historical_evidence.n_cells_significant_worse} "
          f"none={export.historical_evidence.n_cells_no_significant_difference})")
    print(f"prospective_observations: {len(export.prospective_evidence.observations)} "
          f"(resolved_non_replay={export.prospective_evidence.n_resolved_non_replay_observations} "
          f"pending={export.prospective_evidence.n_pending_observations})")
    print(f"research_ledger_entries: {len(export.research_ledger.entries)}")
    print(f"promotion_gates: {len(export.promotion_gates.gates)} "
          f"(amc_promotion_authorized={export.promotion_gates.amc_promotion_authorized})")
    print(f"charts: {len(export.charts)}")
    print(f"export_sha256: {export.export_sha256}")


if __name__ == "__main__":
    main()
