"""Ledger-specific composition of the generic checkpoint/anchor primitive
(P0-3). `verify_ledger_completeness` composes internal chain verification
(`ledger.verifier.verify_chain`) with checkpoint-anchored completeness
(`anchor.verifier.verify_against_checkpoint`) and returns BOTH results,
never conflating them — AT-01/AT-02's updated claims (see
reports/MODEL1_REVIEW_REMEDIATION.md) distinguish "internal chain integrity"
from "checkpoint-anchored completeness" exactly because one does not imply
the other (the review's point C.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_lab.anchor.builder import build_signed_checkpoint
from model_lab.anchor.contracts import SignedCheckpoint, SubjectKind
from model_lab.anchor.storage import append_checkpoint, latest_checkpoint
from model_lab.anchor.verifier import AnchorVerificationResult, verify_against_checkpoint
from model_lab.contracts.ledger_record import LedgerRecord
from model_lab.governance.signer_registry import SignerRegistry
from model_lab.ledger.chain import chain_hash
from model_lab.ledger.storage import LedgerFile
from model_lab.ledger.verifier import VerificationResult, verify_chain


def create_ledger_checkpoint(ledger_path: Path, *, namespace: str, private_key_path: Path, signer_id: str) -> SignedCheckpoint:
    records = LedgerFile(ledger_path).read_all()
    if not records:
        raise ValueError("cannot checkpoint an empty ledger")
    last = records[-1]
    signed = build_signed_checkpoint(
        subject_kind=SubjectKind.LEDGER,
        subject_id=str(ledger_path),
        namespace=namespace,
        anchored_seq=last.seq,
        anchored_head_hash=chain_hash(last),
        anchored_record_count=len(records),
        private_key_path=private_key_path,
        signer_id=signer_id,
    )
    append_checkpoint(ledger_path, signed)
    return signed


@dataclass(frozen=True)
class LedgerCompletenessResult:
    internal: VerificationResult
    anchor: AnchorVerificationResult


def verify_ledger_completeness(
    ledger_path: Path,
    *,
    signer_registry: SignerRegistry,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> LedgerCompletenessResult:
    records: list[LedgerRecord] = LedgerFile(ledger_path).read_all()
    internal = verify_chain(records)

    def _hash_at(seq: int) -> str | None:
        if seq < 0 or seq >= len(records):
            return None
        return chain_hash(records[seq])

    checkpoint = latest_checkpoint(ledger_path)
    anchor = verify_against_checkpoint(
        signed_checkpoint=checkpoint,
        current_count=len(records),
        hash_at_position=_hash_at,
        signer_registry=signer_registry,
        allowed_signers_path=allowed_signers_path,
        skip_crypto_verification=skip_crypto_verification,
    )
    return LedgerCompletenessResult(internal=internal, anchor=anchor)
