"""Checkpoint verification — the four states required by P0-3 remediation:
"chain internally valid but UNANCHORED / chain valid against checkpoint /
tail truncated relative to checkpoint / checkpoint invalid."

This deliberately does NOT re-implement internal chain-integrity checking
(`ledger.verifier.verify_chain` / `options_archive.verifier.verify_archive_chain`
already do that correctly — the review confirmed interior tamper/delete IS
caught). This module answers a narrower, complementary question: is the
CURRENT state consistent with the most recent trusted checkpoint, or has the
tail been truncated relative to it? Callers run both checks and combine them
(see `ledger/anchor.py` / `options_archive/anchor.py` for the composed
API each subject actually exposes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from model_lab.anchor.builder import checkpoint_signed_payload
from model_lab.anchor.contracts import AnchorStatus, SignedCheckpoint
from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.governance.signer_registry import SignerRegistry, SignerRole
from model_lab.ledger.signing import verify_signature


@dataclass(frozen=True)
class AnchorVerificationResult:
    status: AnchorStatus
    detail: str


def verify_checkpoint_signature(
    signed: SignedCheckpoint,
    *,
    signer_registry: SignerRegistry,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> bool:
    """A checkpoint is valid only if: its own self-referential hash
    recomputes, AND at least one attached signature is (a) cryptographically
    valid over that hash, in the checkpoint's namespace, and (b) from a
    principal the SignerRegistry confirms holds the OPERATOR role (P0-3
    requirement 2, reusing the P0-2 identity-binding fix rather than
    re-trusting a self-declared role)."""
    checkpoint = signed.checkpoint
    recomputed = canonical_sha256_excluding(checkpoint.canonical_dict(), "checkpoint_sha256")
    if recomputed != checkpoint.checkpoint_sha256:
        return False

    payload = checkpoint_signed_payload(checkpoint)
    for sig in signed.signatures:
        if sig.namespace != checkpoint.namespace:
            continue
        if not signer_registry.has_role(sig.signer_id, SignerRole.OPERATOR, namespace=sig.namespace):
            continue
        if skip_crypto_verification:
            return True
        if verify_signature(
            payload, sig.signature_b64, allowed_signers_path=allowed_signers_path,
            principal=sig.signer_id, namespace=sig.namespace,
        ):
            return True
    return False


def verify_against_checkpoint(
    *,
    signed_checkpoint: SignedCheckpoint | None,
    current_count: int,
    hash_at_position: Callable[[int], str | None],
    signer_registry: SignerRegistry,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> AnchorVerificationResult:
    """`hash_at_position(seq)` returns the position-hash (chain_hash /
    record_hash) of the record at that seq in the CURRENT state, or None if
    it doesn't exist. For non-sequential subjects (a single publication
    manifest), pass `hash_at_position = lambda _seq: current_manifest_hash`
    and a checkpoint with `anchored_seq=0`.
    """
    if signed_checkpoint is None:
        return AnchorVerificationResult(
            AnchorStatus.UNANCHORED,
            "no checkpoint available -- internal chain integrity only, completeness is NOT verifiable",
        )

    if not verify_checkpoint_signature(
        signed_checkpoint, signer_registry=signer_registry, allowed_signers_path=allowed_signers_path,
        skip_crypto_verification=skip_crypto_verification,
    ):
        return AnchorVerificationResult(
            AnchorStatus.CHECKPOINT_INVALID,
            "checkpoint self-hash or signature does not verify (tampered/forged checkpoint, or no "
            "registry-confirmed operator signature present)",
        )

    checkpoint = signed_checkpoint.checkpoint
    anchored_seq = checkpoint.anchored_seq if checkpoint.anchored_seq is not None else 0

    if current_count <= anchored_seq:
        return AnchorVerificationResult(
            AnchorStatus.TAIL_TRUNCATED,
            f"current record count ({current_count}) is at or below the anchored seq "
            f"({anchored_seq}) -- records present at checkpoint time are now missing",
        )

    current_hash = hash_at_position(anchored_seq)
    if current_hash != checkpoint.anchored_head_hash:
        return AnchorVerificationResult(
            AnchorStatus.TAIL_TRUNCATED,
            f"hash at the anchored position ({current_hash!r}) no longer matches the checkpoint's "
            f"attested hash ({checkpoint.anchored_head_hash!r}) -- history at or before the anchor "
            "was rewritten",
        )

    return AnchorVerificationResult(
        AnchorStatus.ANCHORED_VALID,
        f"current state matches the checkpoint at seq={anchored_seq}; "
        f"{current_count - anchored_seq - 1} record(s) legitimately added since",
    )
