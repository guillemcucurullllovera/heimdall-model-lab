"""Publication manifest authenticity — P1-1 remediation (adversarial review
finding D.1): "`verify_publication` re-hashes every file listed in
`manifest.json` against that same `manifest.json`'s own `file_hashes` — but
`manifest.json`'s own bytes are never hashed or anchored anywhere outside
the publish directory... A coordinated edit of a content file *and* its
corresponding `file_hashes` entry together would not be caught."

Fix: reuse the shared checkpoint/anchor primitive (P0-3,
`model_lab.anchor`) to bind `manifest.json`'s OWN bytes (not just what it
claims about other files) to an operator-signed checkpoint. A coordinated
rewrite of a content file and its `file_hashes` entry changes
`manifest.json`'s own bytes, so it no longer matches the checkpoint's
`anchored_head_hash` -- caught as `TAIL_TRUNCATED` by the same generic
verifier used for the ledger and option archive (the "subject" here is just
one file, not a growing sequence, so `anchored_seq=0` always).

A publication with NO checkpoint is `UNANCHORED` -- still usable for
research purposes (nothing here weakens the existing file/manifest
drift-detection `verify_publication` already does), but never treated as
promotion-grade: `is_promotion_grade` below is the single source of truth
for that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_lab.anchor.builder import build_signed_checkpoint
from model_lab.anchor.contracts import AnchorStatus, SignedCheckpoint, SubjectKind
from model_lab.anchor.storage import append_checkpoint, latest_checkpoint
from model_lab.anchor.verifier import AnchorVerificationResult, verify_against_checkpoint
from model_lab.canon.jcs import sha256_bytes
from model_lab.governance.signer_registry import SignerRegistry
from model_lab.publish.verify import PublicationVerificationResult, verify_publication


def create_publication_checkpoint(
    publication_dir: Path, *, namespace: str, private_key_path: Path, signer_id: str
) -> SignedCheckpoint:
    manifest_path = publication_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no manifest.json at {publication_dir} -- build the publication first")
    manifest_hash = sha256_bytes(manifest_path.read_bytes())
    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.PUBLICATION_MANIFEST,
        subject_id=str(manifest_path),
        namespace=namespace,
        anchored_seq=0,
        anchored_head_hash=manifest_hash,
        anchored_record_count=1,
        private_key_path=private_key_path,
        signer_id=signer_id,
    )
    append_checkpoint(manifest_path, signed)
    return signed


@dataclass(frozen=True)
class PublicationAuthenticityResult:
    file_check: PublicationVerificationResult
    anchor: AnchorVerificationResult

    @property
    def is_promotion_grade(self) -> bool:
        """The single source of truth for "is this publication trustworthy
        enough to support a promotion-grade claim." Both conditions are
        required: internal file/manifest consistency AND a valid, matching,
        operator-signed checkpoint over the manifest's own bytes."""
        return self.file_check.ok and self.anchor.status == AnchorStatus.ANCHORED_VALID


def verify_publication_authenticity(
    publication_dir: Path,
    *,
    signer_registry: SignerRegistry,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> PublicationAuthenticityResult:
    file_check = verify_publication(publication_dir)
    manifest_path = publication_dir / "manifest.json"

    def _hash_at(_seq: int) -> str | None:
        if not manifest_path.exists():
            return None
        return sha256_bytes(manifest_path.read_bytes())

    checkpoint = latest_checkpoint(manifest_path)
    current_count = 1 if manifest_path.exists() else 0
    anchor = verify_against_checkpoint(
        signed_checkpoint=checkpoint,
        current_count=current_count,
        hash_at_position=_hash_at,
        signer_registry=signer_registry,
        allowed_signers_path=allowed_signers_path,
        skip_crypto_verification=skip_crypto_verification,
    )
    return PublicationAuthenticityResult(file_check=file_check, anchor=anchor)
