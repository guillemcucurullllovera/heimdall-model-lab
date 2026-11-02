"""Checkpoint storage: a sibling `<subject>.checkpoints.ndjson` file, always
separate from the subject's own mutable data file.

**Why separate, and the honest limitation this still carries today** (P0-3
requirement 8, "explicitly document the future operational requirement"):
storing checkpoints in a different file means an ordinary crash, an
accidental `rm`, or a bug in the subject's own writer cannot silently rewrite
a checkpoint alongside truncating the data file — the two are independent
artifacts. It does NOT yet mean checkpoints are safe from a deliberate,
privileged attacker: today both files live under the same
`heimdall-model-lab-data/` estate, writable by the same Unix account that
writes the data file, with no off-host copy and no real operator key (D2 —
none is generated in this phase). A sufficiently privileged attacker (or
future automated writer with a bug) could still rewrite this file too. The
production requirement this mechanism is built for — "trusted checkpoint
copies/signatures must exist outside the mutable evidence directory /
ordinary writer authority," e.g. git-committed, or held off-host by the
operator, or written by a process with no other write access to the evidence
directory — is NOT satisfied yet and must be before this is treated as a
real defense against a privileged attacker. What IS true today: append-only,
hash-chained, and now checkpoint-anchored verification correctly detects
every truncation/rewrite scenario this session's exploit and regression
tests exercise, under the threat model of "an ordinary writer (bug, crash,
accidental deletion, or a non-privileged tamper) modifies the data file
without ALSO correctly forging a new signed checkpoint."
"""

from __future__ import annotations

from pathlib import Path

from model_lab.anchor.contracts import SignedCheckpoint


def checkpoint_path_for(subject_data_path: Path) -> Path:
    return subject_data_path.with_name(subject_data_path.name + ".checkpoints.ndjson")


def append_checkpoint(subject_data_path: Path, signed: SignedCheckpoint) -> None:
    import os

    path = checkpoint_path_for(subject_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(signed.model_dump_json())
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def read_all_checkpoints(subject_data_path: Path) -> list[SignedCheckpoint]:
    path = checkpoint_path_for(subject_data_path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(SignedCheckpoint.model_validate_json(line))
    return out


def latest_checkpoint(subject_data_path: Path) -> SignedCheckpoint | None:
    """The most recently APPENDED checkpoint (by file order — checkpoints
    are append-only, not re-sorted by anchored_seq, so a well-behaved
    checkpointing process must always append monotonically)."""
    all_cps = read_all_checkpoints(subject_data_path)
    return all_cps[-1] if all_cps else None
