"""Hash-chain verification for the option archive — mirrors
`model_lab.ledger.verifier` but for `OptionSnapshotRecord` (§18a rule 1:
append-only with prev_hash chaining; "the archive can always be re-derived"
from the raw payloads, rule 2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from model_lab.contracts.option_snapshot import OptionSnapshotRecord
from model_lab.options_archive.storage import record_hash_of


@dataclass(frozen=True)
class ArchiveViolation:
    archive_seq: int
    kind: str
    detail: str


@dataclass(frozen=True)
class ArchiveVerificationResult:
    ok: bool
    n_records: int
    violations: list[ArchiveViolation] = field(default_factory=list)


def verify_archive_chain(records: list[OptionSnapshotRecord]) -> ArchiveVerificationResult:
    violations: list[ArchiveViolation] = []
    expected_seq = 0
    expected_prev_hash: str | None = None

    for rec in records:
        if rec.archive_seq != expected_seq:
            violations.append(
                ArchiveViolation(rec.archive_seq, "seq_gap", f"expected {expected_seq}, got {rec.archive_seq}")
            )
        if rec.prev_hash != expected_prev_hash:
            violations.append(
                ArchiveViolation(
                    rec.archive_seq,
                    "prev_hash_mismatch",
                    f"expected {expected_prev_hash!r}, got {rec.prev_hash!r}",
                )
            )
        recomputed = record_hash_of(rec)
        if recomputed != rec.record_hash:
            violations.append(
                ArchiveViolation(
                    rec.archive_seq,
                    "record_hash_mismatch",
                    f"stored={rec.record_hash!r} recomputed={recomputed!r}",
                )
            )

        expected_seq = rec.archive_seq + 1
        expected_prev_hash = rec.record_hash  # chain forward from the RECOMPUTED value would be better;
        # kept as stored-value chaining here (see ledger/verifier.py for why using the record's own
        # stored fields, not a recomputed substitute, is the simpler and still-correct approach: any
        # inconsistency is already caught by the record_hash_mismatch check above, and a downstream
        # re-chain off a tampered-but-hash-updated record is caught by the next record's prev_hash_mismatch).

    return ArchiveVerificationResult(ok=not violations, n_records=len(records), violations=violations)
