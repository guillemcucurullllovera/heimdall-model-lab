"""Ledger chain verification — AT-01 (tamper detection) and the general
"verify-on-load" rule from MODEL1_foundation_contracts.md §12.

Verification recomputes, for every record in sequence: seq monotonicity,
prev_hash == chain_hash(previous record), body_hash == canonical_sha256(body).
Any mismatch is a violation; verification does not stop at the first one — it
reports every record from that point forward as unverifiable (their prev_hash
values are only meaningful relative to a chain that no longer recomputes),
which is what lets `lineages_to_quarantine` identify every ModelVersion
record downstream of a tamper point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_lab.canon.jcs import canonical_sha256
from model_lab.contracts.ledger_record import LedgerRecord, LedgerRecordType
from model_lab.ledger.chain import chain_hash


@dataclass(frozen=True)
class Violation:
    seq: int
    kind: str  # "seq_gap" | "prev_hash_mismatch" | "body_hash_mismatch"
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    n_records: int
    violations: list[Violation]
    first_unverifiable_seq: int | None
    model_versions_to_quarantine: list[str] = field(default_factory=list)


def verify_chain(records: list[LedgerRecord]) -> VerificationResult:
    violations: list[Violation] = []
    expected_seq = 0
    expected_prev_hash: str | None = None
    first_bad: int | None = None

    for rec in records:
        bad_here = False
        if rec.seq != expected_seq:
            violations.append(
                Violation(rec.seq, "seq_gap", f"expected seq={expected_seq}, got {rec.seq}")
            )
            bad_here = True
        if rec.prev_hash != expected_prev_hash:
            violations.append(
                Violation(
                    rec.seq,
                    "prev_hash_mismatch",
                    f"expected prev_hash={expected_prev_hash!r}, got {rec.prev_hash!r}",
                )
            )
            bad_here = True
        recomputed_body_hash = canonical_sha256(rec.body)
        if recomputed_body_hash != rec.body_hash:
            violations.append(
                Violation(
                    rec.seq,
                    "body_hash_mismatch",
                    f"stored body_hash={rec.body_hash!r}, recomputed={recomputed_body_hash!r} "
                    "(record body was altered after being written)",
                )
            )
            bad_here = True

        if bad_here and first_bad is None:
            first_bad = rec.seq

        # Two independent checks compose to make tampering unrecoverable
        # without re-signing the whole downstream chain: (1) body_hash_mismatch
        # above catches an edited body whose body_hash wasn't also updated;
        # (2) chaining forward from this record's *stored* fields (not a
        # recomputed substitute) means that if an attacker DOES update
        # body_hash to match a tampered body, this record's chain_hash
        # changes, so the *next* record's stored prev_hash (fixed at write
        # time, before the tamper) no longer matches — caught as
        # prev_hash_mismatch at seq+1. Only a full, consistent rewrite of
        # every record from the tamper point onward would pass this
        # verifier; that is exactly what the off-host operator-signed
        # checkpoint (head hash committed to git) is the final defense
        # against, per §12.
        expected_seq = rec.seq + 1
        expected_prev_hash = chain_hash(rec)

    quarantine = _model_versions_at_or_after(records, first_bad) if first_bad is not None else []

    return VerificationResult(
        ok=not violations,
        n_records=len(records),
        violations=violations,
        first_unverifiable_seq=first_bad,
        model_versions_to_quarantine=quarantine,
    )


def _model_versions_at_or_after(records: list[LedgerRecord], first_bad_seq: int) -> list[str]:
    """Every ModelVersion record at or after the first tampered/corrupt
    record, PLUS every later ModelVersion whose predecessor_hash chain
    passes through one of those — i.e. the whole downstream lineage — must
    be quarantined (§12: "a tamper test ... must fail verification and
    quarantine every model whose lineage passes through it")."""
    tainted_hashes: set[str] = set()
    quarantine_ids: list[str] = []
    for rec in records:
        if rec.record_type != LedgerRecordType.MODEL_VERSION:
            continue
        body = rec.body
        predecessor = body.get("predecessor_hash")
        model_version_hash = body.get("model_version_hash")
        tainted = rec.seq >= first_bad_seq or (predecessor is not None and predecessor in tainted_hashes)
        if tainted:
            if model_version_hash:
                tainted_hashes.add(model_version_hash)
            model_id = body.get("model_id") or model_version_hash
            if model_id:
                quarantine_ids.append(model_id)
    return quarantine_ids
