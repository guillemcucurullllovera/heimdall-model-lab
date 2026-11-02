"""Option-archive composition of the generic checkpoint/anchor primitive
(P0-3 requirement 6: "Apply equivalent protection to the option archive or
factor out a shared checkpoint primitive if scientifically appropriate" --
we factored out the shared primitive, model_lab.anchor; this module is the
thin composition, mirroring ledger/anchor.py exactly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_lab.anchor.builder import build_signed_checkpoint
from model_lab.anchor.contracts import SignedCheckpoint, SubjectKind
from model_lab.anchor.storage import append_checkpoint, latest_checkpoint
from model_lab.anchor.verifier import AnchorVerificationResult, verify_against_checkpoint
from model_lab.contracts.option_snapshot import OptionSnapshotRecord
from model_lab.governance.signer_registry import SignerRegistry
from model_lab.options_archive.storage import OptionArchiveFile
from model_lab.options_archive.verifier import ArchiveVerificationResult, verify_archive_chain


def create_archive_checkpoint(archive_path: Path, *, namespace: str, private_key_path: Path, signer_id: str) -> SignedCheckpoint:
    records = OptionArchiveFile(archive_path).read_all()
    if not records:
        raise ValueError("cannot checkpoint an empty option archive")
    last = records[-1]
    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.OPTION_ARCHIVE,
        subject_id=str(archive_path),
        namespace=namespace,
        anchored_seq=last.archive_seq,
        anchored_head_hash=last.record_hash,
        anchored_record_count=len(records),
        private_key_path=private_key_path,
        signer_id=signer_id,
    )
    append_checkpoint(archive_path, signed)
    return signed


@dataclass(frozen=True)
class ArchiveCompletenessResult:
    internal: ArchiveVerificationResult
    anchor: AnchorVerificationResult


def verify_archive_completeness(
    archive_path: Path,
    *,
    signer_registry: SignerRegistry,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> ArchiveCompletenessResult:
    records: list[OptionSnapshotRecord] = OptionArchiveFile(archive_path).read_all()
    internal = verify_archive_chain(records)

    def _hash_at(seq: int) -> str | None:
        if seq < 0 or seq >= len(records):
            return None
        return records[seq].record_hash

    checkpoint = latest_checkpoint(archive_path)
    anchor = verify_against_checkpoint(
        signed_checkpoint=checkpoint,
        current_count=len(records),
        hash_at_position=_hash_at,
        signer_registry=signer_registry,
        allowed_signers_path=allowed_signers_path,
        skip_crypto_verification=skip_crypto_verification,
    )
    return ArchiveCompletenessResult(internal=internal, anchor=anchor)
