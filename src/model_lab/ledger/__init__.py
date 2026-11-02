from model_lab.ledger.chain import chain_hash, compute_body_hash
from model_lab.ledger.signing import (
    SignatureVerificationError,
    SigningToolUnavailableError,
    ThrowawayKeypair,
    find_principals,
    generate_throwaway_keypair,
    sign_bytes,
    verify_signature,
    write_allowed_signers,
)
from model_lab.ledger.storage import (
    CorruptLedgerError,
    LedgerFile,
    TornTailRepair,
    read_raw_lines_with_torn_tail_check,
    repair_torn_tail,
)
from model_lab.ledger.verifier import VerificationResult, Violation, verify_chain
from model_lab.ledger.writer import LedgerWriter, PreAppendCheck

# NOTE: `ledger.anchor` (checkpoint/completeness composition) is deliberately
# NOT imported here. model_lab.anchor (the generic checkpoint primitive)
# itself depends on model_lab.ledger.signing, so model_lab.ledger <->
# model_lab.anchor is a genuine bidirectional package dependency -- eagerly
# importing ledger.anchor from this __init__.py creates a real circular
# import the first time something imports model_lab.anchor before
# model_lab.ledger (confirmed by testing both import orders). Import it
# explicitly instead: `from model_lab.ledger.anchor import verify_ledger_completeness`.

__all__ = [
    "chain_hash",
    "compute_body_hash",
    "LedgerFile",
    "TornTailRepair",
    "CorruptLedgerError",
    "read_raw_lines_with_torn_tail_check",
    "repair_torn_tail",
    "LedgerWriter",
    "PreAppendCheck",
    "verify_chain",
    "VerificationResult",
    "Violation",
    "generate_throwaway_keypair",
    "ThrowawayKeypair",
    "sign_bytes",
    "verify_signature",
    "write_allowed_signers",
    "find_principals",
    "SigningToolUnavailableError",
    "SignatureVerificationError",
]
