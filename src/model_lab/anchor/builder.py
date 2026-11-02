"""Build and sign a Checkpoint. No live/off-host operator key exists yet (D2)
— production checkpoints are signed out-of-band by the operator, exactly
like every other signature in this codebase; tests use throwaway keys."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.anchor.contracts import Checkpoint, SignedCheckpoint, SubjectKind
from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.contracts.ledger_record import Signature
from model_lab.ledger.signing import sign_bytes


def build_checkpoint(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    namespace: str,
    anchored_seq: int | None,
    anchored_head_hash: str,
    anchored_record_count: int | None = None,
    now: datetime | None = None,
) -> Checkpoint:
    now = now or datetime.now(UTC)
    draft = Checkpoint(
        subject_kind=subject_kind,
        subject_id=subject_id,
        namespace=namespace,
        anchored_seq=anchored_seq,
        anchored_head_hash=anchored_head_hash,
        anchored_record_count=anchored_record_count,
        created_at_utc=now,
        checkpoint_sha256="0" * 64,  # placeholder
    )
    real_hash = canonical_sha256_excluding(draft.canonical_dict(), "checkpoint_sha256")
    return draft.model_copy(update={"checkpoint_sha256": real_hash})


def checkpoint_signed_payload(checkpoint: Checkpoint) -> bytes:
    """The bytes a checkpoint signature is over: its own self-referential
    hash, matching the convention used for TransitionRequest/v1-freeze
    signing elsewhere in this codebase."""
    return checkpoint.checkpoint_sha256.encode("utf-8")


def sign_checkpoint(checkpoint: Checkpoint, *, private_key_path: Path, signer_id: str, signer_role: str = "operator") -> Signature:
    """Sign with a real key (throwaway in tests). Production callers use an
    operator-held, off-host key; MODEL-1 never generates one itself."""
    payload = checkpoint_signed_payload(checkpoint)
    sig_text = sign_bytes(payload, private_key_path, checkpoint.namespace)
    return Signature(
        signer_role=signer_role,
        signer_id=signer_id,
        namespace=checkpoint.namespace,
        signature_b64=sig_text,
        signed_at_utc=datetime.now(UTC),
    )


def build_signed_checkpoint(
    *,
    subject_kind: SubjectKind,
    subject_id: str,
    namespace: str,
    anchored_seq: int | None,
    anchored_head_hash: str,
    anchored_record_count: int | None,
    private_key_path: Path,
    signer_id: str,
) -> SignedCheckpoint:
    checkpoint = build_checkpoint(
        subject_kind=subject_kind,
        subject_id=subject_id,
        namespace=namespace,
        anchored_seq=anchored_seq,
        anchored_head_hash=anchored_head_hash,
        anchored_record_count=anchored_record_count,
    )
    sig = sign_checkpoint(checkpoint, private_key_path=private_key_path, signer_id=signer_id)
    return SignedCheckpoint(checkpoint=checkpoint, signatures=[sig])
