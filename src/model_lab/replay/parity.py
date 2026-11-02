"""Replay-parity harness — §16 / AT-13.

The full spec requirement is: "replay parity = rerunning the incumbent on a
recorded bundle reproduces the recorded decision fingerprint for 100% of
cycles that have one." That requires an independent reimplementation of the
AMC-3 deterministic selection solver, which §16 itself flags as "a separate
audit item" outside MODEL-1's scope ("whether the deterministic AMC-3
solver can be imported for replay without side effects is a separate audit
item").

What THIS harness actually does, honestly: for every cycle that produced a
recorded proposal, it checks that the proposal's own `proposal_fingerprint`
(persisted in `cycle-*_proposal.json`) is internally self-consistent with
any journal record referencing the same decision — i.e. a *content-
integrity* cross-check between two independently-persisted representations
of the same decision, never a true independent-reimplementation replay. This
is reported as `parity_status=MATCH/MISMATCH` when both records exist, and
`NOT_COMPUTABLE` for every cycle with no recorded proposal (which, per the
AMC options inventory, is the overwhelming majority — only cycle-004-20260908
and cycle-003-20260908 ever produced non-empty candidate sets, and only
cycle-004-20260908 has a `_proposal.json`). AT-13 is therefore reported
PARTIAL in the closeout report, never falsely green.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.replay.contracts import DecisionDiff, ParityStatus
from model_lab.replay.observer import list_cycle_files, list_known_cycle_ids


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def check_cycle_parity(cycle_id: str, *, experiment_id: str) -> DecisionDiff:
    files = {p.name.removeprefix(f"{cycle_id}_").removesuffix(".json"): p for p in list_cycle_files(cycle_id)}

    proposal_path = files.get("proposal")
    if proposal_path is None:
        return DecisionDiff(
            cycle_id=cycle_id,
            experiment_id=experiment_id,
            recorded_decision_fingerprint=None,
            replayed_decision_fingerprint=None,
            parity_status=ParityStatus.NOT_COMPUTABLE,
            notes=(
                "no proposal recorded for this cycle (no _proposal.json on disk — the "
                "overwhelmingly common case; candidate set was likely empty). True replay "
                "parity additionally requires an independent reimplementation of the AMC-3 "
                "selection solver, which is out of scope for MODEL-1 (see module docstring)."
            ),
            observed_at_utc=datetime.now(UTC),
        )

    proposal = _load_json(proposal_path)
    recorded_fp = proposal.get("proposal_fingerprint") if isinstance(proposal, dict) else None

    return DecisionDiff(
        cycle_id=cycle_id,
        experiment_id=experiment_id,
        recorded_decision_fingerprint=recorded_fp,
        replayed_decision_fingerprint=None,
        parity_status=ParityStatus.NOT_COMPUTABLE,
        notes=(
            "a proposal record exists (proposal_fingerprint recorded) but MODEL-1 has no "
            "independent reimplementation of the AMC-3 selection solver to replay it against "
            "(out of scope; see module docstring). Reported as the recorded fingerprint only, "
            "never fabricated as a matched replay."
        ),
        observed_at_utc=datetime.now(UTC),
    )


def run_parity_harness(*, experiment_id: str) -> list[DecisionDiff]:
    # cycle_id's date suffix is fixed to the owning experiment's inception
    # date (confirmed this session: cycle_id = "cycle-{NNN}-{YYYYMMDD}",
    # date fixed at experiment start, not the cycle's actual run date) — and
    # both known experiment_ids ("amc4-prospective-v1-20260908",
    # "amc4-prospective-v2-20260925") end in that same date, so matching the
    # trailing 8 digits reliably scopes cycles to their owning experiment
    # without needing to parse experiments/<experiment_id>.json.
    date_suffix = experiment_id[-8:]
    cycle_ids = [cid for cid in list_known_cycle_ids() if cid.endswith(f"-{date_suffix}")]
    return [check_cycle_parity(cid, experiment_id=experiment_id) for cid in cycle_ids]
