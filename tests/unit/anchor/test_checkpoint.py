from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from model_lab.anchor.builder import build_checkpoint, build_signed_checkpoint
from model_lab.anchor.contracts import AnchorStatus, SignedCheckpoint, SubjectKind
from model_lab.anchor.storage import append_checkpoint, latest_checkpoint, read_all_checkpoints
from model_lab.anchor.verifier import verify_against_checkpoint, verify_checkpoint_signature
from model_lab.contracts.ledger_record import Signature
from model_lab.governance.signer_registry import RegisteredSigner, SignerRegistry, SignerRole
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    write_allowed_signers,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _registry_with_operator(principal: str) -> SignerRegistry:
    return SignerRegistry(
        signers=[
            RegisteredSigner(
                principal=principal, roles=[SignerRole.OPERATOR], namespaces=["ns"], public_key_line="fake"
            )
        ]
    )


def test_build_checkpoint_self_hash_is_stable() -> None:
    cp1 = build_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=5,
        anchored_head_hash="a" * 64, anchored_record_count=6, now=NOW,
    )
    cp2 = build_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=5,
        anchored_head_hash="a" * 64, anchored_record_count=6, now=NOW,
    )
    assert cp1.checkpoint_sha256 == cp2.checkpoint_sha256


def test_build_checkpoint_hash_changes_with_content() -> None:
    cp1 = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=5, anchored_head_hash="a" * 64, anchored_record_count=6, now=NOW)
    cp2 = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=6, anchored_head_hash="a" * 64, anchored_record_count=7, now=NOW)
    assert cp1.checkpoint_sha256 != cp2.checkpoint_sha256


def test_checkpoint_storage_append_and_read_round_trip(tmp_path: Path) -> None:
    subject = tmp_path / "ledger.ndjson"
    try:
        key = generate_throwaway_keypair(tmp_path / "op-1")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id=str(subject), namespace="ns",
        anchored_seq=2, anchored_head_hash="b" * 64, anchored_record_count=3,
        private_key_path=key.private_key_path, signer_id="op-1",
    )
    assert read_all_checkpoints(subject) == []
    append_checkpoint(subject, signed)
    all_cps = read_all_checkpoints(subject)
    assert len(all_cps) == 1
    loaded_latest = latest_checkpoint(subject)
    assert loaded_latest is not None
    assert loaded_latest.checkpoint.anchored_seq == 2


def test_verify_checkpoint_signature_valid(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "op-1")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("op-1", ["ns"], key.public_key_text)])
    registry = _registry_with_operator("op-1")

    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns",
        anchored_seq=1, anchored_head_hash="c" * 64, anchored_record_count=2,
        private_key_path=key.private_key_path, signer_id="op-1",
    )
    assert verify_checkpoint_signature(signed, signer_registry=registry, allowed_signers_path=allowed_signers)


def test_verify_checkpoint_signature_rejects_unregistered_signer(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "not-operator")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("not-operator", ["ns"], key.public_key_text)])
    registry = SignerRegistry(signers=[])  # nobody registered

    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns",
        anchored_seq=1, anchored_head_hash="c" * 64, anchored_record_count=2,
        private_key_path=key.private_key_path, signer_id="not-operator",
    )
    assert not verify_checkpoint_signature(signed, signer_registry=registry, allowed_signers_path=allowed_signers)


def test_verify_checkpoint_signature_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "op-1")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("op-1", ["ns"], key.public_key_text)])
    registry = _registry_with_operator("op-1")

    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns",
        anchored_seq=1, anchored_head_hash="c" * 64, anchored_record_count=2,
        private_key_path=key.private_key_path, signer_id="op-1",
    )
    tampered = signed.model_copy(update={"checkpoint": signed.checkpoint.model_copy(update={"anchored_seq": 999})})
    assert not verify_checkpoint_signature(tampered, signer_registry=registry, allowed_signers_path=allowed_signers)


# --- verify_against_checkpoint: the four required states -------------------


def test_unanchored_when_no_checkpoint_exists() -> None:
    result = verify_against_checkpoint(
        signed_checkpoint=None, current_count=5, hash_at_position=lambda s: "x",
        signer_registry=SignerRegistry(signers=[]), allowed_signers_path=Path("/dev/null"),
    )
    assert result.status == AnchorStatus.UNANCHORED


def test_anchored_valid_when_position_hash_matches() -> None:
    registry = _registry_with_operator("op-1")
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=2, anchored_head_hash="2" * 64, anchored_record_count=3, now=NOW)

    signed = SignedCheckpoint(
        checkpoint=cp,
        signatures=[Signature(signer_role="operator", signer_id="op-1", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    positions = {0: "0" * 64, 1: "1" * 64, 2: "2" * 64, 3: "3" * 64, 4: "4" * 64}
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=5, hash_at_position=lambda s: positions.get(s),
        signer_registry=registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.ANCHORED_VALID


def test_tail_truncated_when_count_drops_below_anchor() -> None:
    registry = _registry_with_operator("op-1")
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=9, anchored_head_hash="9" * 64, anchored_record_count=10, now=NOW)

    signed = SignedCheckpoint(
        checkpoint=cp,
        signatures=[Signature(signer_role="operator", signer_id="op-1", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=3, hash_at_position=lambda s: None,
        signer_registry=registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.TAIL_TRUNCATED


def test_tail_truncated_when_position_hash_no_longer_matches() -> None:
    """History was rewritten at/before the anchor -- the anchored seq exists
    but now has a DIFFERENT hash than what was checkpointed."""
    registry = _registry_with_operator("op-1")
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=2, anchored_head_hash="a" * 64, anchored_record_count=5, now=NOW)

    signed = SignedCheckpoint(
        checkpoint=cp,
        signatures=[Signature(signer_role="operator", signer_id="op-1", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=6, hash_at_position=lambda s: "b" * 64 if s == 2 else "c" * 64,
        signer_registry=registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.TAIL_TRUNCATED


def test_checkpoint_invalid_for_forged_checkpoint() -> None:
    registry = _registry_with_operator("op-1")
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=2, anchored_head_hash="2" * 64, anchored_record_count=3, now=NOW)
    forged = cp.model_copy(update={"anchored_seq": 999})  # checkpoint_sha256 no longer matches content

    signed = SignedCheckpoint(
        checkpoint=forged,
        signatures=[Signature(signer_role="operator", signer_id="op-1", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=5, hash_at_position=lambda s: "2" * 64,
        signer_registry=registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.CHECKPOINT_INVALID


def test_checkpoint_invalid_when_no_registry_confirmed_operator_signs() -> None:
    empty_registry = SignerRegistry(signers=[])
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=2, anchored_head_hash="2" * 64, anchored_record_count=3, now=NOW)

    signed = SignedCheckpoint(
        checkpoint=cp,
        signatures=[Signature(signer_role="operator", signer_id="unregistered", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=5, hash_at_position=lambda s: "2" * 64,
        signer_registry=empty_registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.CHECKPOINT_INVALID


def test_anchored_valid_permits_legitimate_growth_after_the_checkpoint() -> None:
    """Records legitimately added AFTER the checkpoint (new growth) must NOT
    be reported as truncation -- only a MISMATCH at or before the anchor is."""
    registry = _registry_with_operator("op-1")
    cp = build_checkpoint(subject_kind=SubjectKind.LEDGER, subject_id="x", namespace="ns", anchored_seq=1, anchored_head_hash="1" * 64, anchored_record_count=2, now=NOW)

    signed = SignedCheckpoint(
        checkpoint=cp,
        signatures=[Signature(signer_role="operator", signer_id="op-1", namespace="ns", signature_b64="x", signed_at_utc=NOW)],
    )
    positions = {0: "0" * 64, 1: "1" * 64, 2: "2" * 64, 3: "3" * 64, 4: "4" * 64, 5: "5" * 64, 6: "6" * 64, 7: "7" * 64, 8: "8" * 64, 9: "9" * 64}
    result = verify_against_checkpoint(
        signed_checkpoint=signed, current_count=10, hash_at_position=lambda s: positions.get(s),
        signer_registry=registry, allowed_signers_path=Path("/dev/null"), skip_crypto_verification=True,
    )
    assert result.status == AnchorStatus.ANCHORED_VALID
