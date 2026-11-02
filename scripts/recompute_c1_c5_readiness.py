#!/usr/bin/env python3
"""Stage 13 of the build: mechanically rerun task-registry status derivation
against the real data now on disk.

## P0-1 remediation (adversarial review finding C.1)

The original version of this script hand-typed `research_state=...` /
`status=...` literals directly onto `DataRequirement` copies -- exactly the
"trusted metadata instead of a real mechanical connection" pattern the
review flagged (and the same root cause as finding C.1's Task-level
firewall gap). That pattern is now IMPOSSIBLE: `Task`/`DataRequirement`
construction requires a real `DatasetSnapshotRegistry` and fails closed on
any claim that doesn't match a real `DatasetSnapshot` (contracts/task.py).

This script therefore no longer simulates an "after" state by hand -- it
simply calls `build_default_task_registry()` (which does all registry
resolution internally, mechanically, against the real on-disk data estate)
and compares it against the frozen "before" baseline: every task was
BLOCKED on 2026-09-30, before any data bootstrap had run (the very first
fact this whole project established). No `DataRequirement` object is
constructed here at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.task import PromotionClass, TaskStatus
from model_lab.tasks.registry import build_default_task_registry

OUT_JSON = Path(__file__).resolve().parents[1] / "reports" / "c1_c5_updated_readiness.json"

# The historical baseline: every task was BLOCKED on 2026-09-30 before any
# data bootstrap occurred (MODEL1_READINESS_PACK.md §4). This is a frozen
# historical fact, not a live computation -- there is no registry to
# recompute it against (no DatasetSnapshots existed at all at that point).
BASELINE_STATUS = TaskStatus.BLOCKED
BASELINE_PROMOTION_CLASS = PromotionClass.BLOCKED


def main() -> None:
    after = build_default_task_registry()

    rows = []
    for task_id, task in after.items():
        changed = (task.status, task.promotion_class) != (BASELINE_STATUS, BASELINE_PROMOTION_CLASS)
        rows.append(
            {
                "task_id": task_id,
                "status_before": BASELINE_STATUS,
                "status_after": task.status,
                "promotion_class_before": BASELINE_PROMOTION_CLASS,
                "promotion_class_after": task.promotion_class,
                "changed": changed,
            }
        )
        print(
            f"{task_id:12s} status {BASELINE_STATUS:>10s} -> {task.status:<10s}  "
            f"promotion_class {BASELINE_PROMOTION_CLASS:>18s} -> {task.promotion_class:<18s}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
