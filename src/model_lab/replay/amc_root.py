"""The pinned, READ-ONLY AMC root (D5: `./local-data/replay-input`,
experiment `amc4-prospective-v2-20260925`). Every function in `replay/` and
`options_archive/` that touches this path opens files in read mode only —
nothing in MODEL-1 ever writes here, runs an AMC CLI mode against it, or
uses `command_center_state.json` or the stray nested `amc4/amc4/` duplicate
worktree copy as evidence (both explicitly excluded per D5).
"""

from __future__ import annotations

from pathlib import Path

AMC_ROOT = Path("./local-data/replay-input")
JOURNAL_PATH = AMC_ROOT / "amc4_journal.ndjson"
CYCLES_DIR = AMC_ROOT / "cycles"
LEDGER_DIR = AMC_ROOT / "ledger"
EXPERIMENTS_DIR = AMC_ROOT / "experiments"
PENDING_DIR = AMC_ROOT / "pending"

# Explicitly excluded from evidence, per D5.
EXCLUDED_PATHS = frozenset(
    {
        AMC_ROOT / "command_center_state.json",
        AMC_ROOT / "amc4",  # the stray nested duplicate worktree copy
    }
)

PINNED_EXPERIMENT_ID = "amc4-prospective-v2-20260925"
