#!/usr/bin/env python3
"""Remediation (remediation-specification item 5 / independent review
P1-E): "extend MODEL-2 evidence into the existing ledger / RunManifest
provenance system, or explicitly document the lighter-weight status...
Prefer the ledger integration." This script does the former, for the one
place MODEL-2 produces governance-shaped records today (the shadow
`ModelVersion`/`ModelOutput` pairs built by `shadow/build.py`): it appends
one real, hash-chained `LedgerRecord` (record_type=`SHADOW_RECORD`, the
type MODEL-1's own contract already reserves for exactly this -- see
`contracts/ledger_record.py` -- and that no code in this repository had
ever actually used before this remediation) per shadow record, using the
real `LedgerWriter`/`LedgerFile` machinery MODEL-1 built and tests (AT-01,
AT-02, AT-03), not an ad hoc append.

Deliberately writes to a LEDGER FILE LOCAL TO THIS GIT WORKTREE
(`reports/ledger/model2_shadow_ledger.ndjson`), not into MODEL-1's own
production evidence ledger under the shared, non-git data estate
(`./local-data/`). That directory is shared
across every worktree of this repository on disk (confirmed: it is NOT
worktree-scoped), so writing into it from a remediation pass in a review
worktree would be a cross-worktree side effect outside this task's
boundary ("do not modify ... other production boundaries"). A real,
hash-chained, tamper-evident, append-only ledger -- using the SAME
contracts and writer MODEL-1 itself uses -- is still what this script
produces; it is just scoped to this worktree's own `reports/`, exactly
like every other MODEL-2 evidence artifact. Extending this to MODEL-1's
shared production ledger (if ever desired) is a separate, explicit,
operator-authorised step, not something a remediation pass should do
unilaterally.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.ledger.writer import LedgerWriter

LEDGER_PATH = Path(__file__).resolve().parents[1] / "reports" / "ledger" / "model2_shadow_ledger.ndjson"
SNAPSHOT_GLOB = "snapshot_*.json"
SHADOW_DIR = Path(__file__).resolve().parents[1] / "reports" / "shadow"


def _stringify_floats(obj: object) -> object:
    """MODEL-1's own canonical-JSON (JCS) hasher refuses any non-integral
    float (`canon/jcs.py::NonCanonicalFloatError`) by design -- "represent
    this value as a string instead (MODEL-1's own money/precision
    convention)," already followed by `ModelOutput.point_estimate`/
    `QuantilePoint.value`. One field in the existing, frozen `ModelOutput`
    contract does NOT follow it: `QuantilePoint.quantile` is a `float`
    (e.g. `0.95`), which is why no record with quantiles had ever been run
    through the ledger before this script. Rather than changing a frozen
    contract as a side effect of adding ledger anchoring, this recursively
    stringifies any non-integral float found in the ledger BODY payload
    only (the contract objects themselves, and the separately-written
    `*_params.json` artifacts, are untouched)."""
    if isinstance(obj, dict):
        return {k: _stringify_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_floats(v) for v in obj]
    if isinstance(obj, float) and not obj.is_integer():
        return str(obj)
    return obj


def _latest_snapshot() -> Path:
    candidates = sorted(SHADOW_DIR.glob(SNAPSHOT_GLOB))
    if not candidates:
        raise FileNotFoundError(f"no shadow snapshot found under {SHADOW_DIR}")
    return candidates[-1]


def main() -> None:
    snapshot_path = _latest_snapshot()
    snapshot = json.loads(snapshot_path.read_text())
    writer = LedgerWriter(LEDGER_PATH, author_id="model2_shadow_ledger_anchor_script")

    appended = 0
    for ticker, entry in snapshot["records"].items():
        # Remediation (2026-10-02 Phase 2 / independent review P2-5):
        # "regime_2state"/"changepoint_bocpd" added alongside the existing
        # five per-ticker T-VOL/T-TAIL family keys.
        for family_key in ("vol", "tail", "ewma", "egarch_t", "combo_median", "regime_2state", "changepoint_bocpd"):
            if family_key not in entry:
                continue
            family = entry[family_key]
            if "model_version" not in family:  # SKIPPED entry, nothing to anchor
                continue
            body = _stringify_floats({
                "ticker": ticker,
                "family_key": family_key,
                "snapshot_source_file": str(snapshot_path.relative_to(LEDGER_PATH.parents[2])),
                "model_version": family["model_version"],
                "model_output": family["model_output"],
            })
            record = writer.append(LedgerRecordType.SHADOW_RECORD, body)
            appended += 1
            print(f"anchored {ticker}/{family_key} as ledger seq={record.seq}")

    # Remediation (2026-10-02 Phase 2 / independent review P2-5): T-ANOM's
    # panel-level record lives under "panel_records", not "records" --
    # anchor it the same way, with "panel" standing in for a per-ticker key.
    for family_key, family in snapshot.get("panel_records", {}).items():
        if "model_version" not in family:
            continue
        body = _stringify_floats({
            "ticker": "panel",
            "family_key": family_key,
            "snapshot_source_file": str(snapshot_path.relative_to(LEDGER_PATH.parents[2])),
            "model_version": family["model_version"],
            "model_output": family["model_output"],
        })
        record = writer.append(LedgerRecordType.SHADOW_RECORD, body)
        appended += 1
        print(f"anchored panel/{family_key} as ledger seq={record.seq}")

    print(f"\nAppended {appended} SHADOW_RECORD entries to {LEDGER_PATH}")


if __name__ == "__main__":
    main()
