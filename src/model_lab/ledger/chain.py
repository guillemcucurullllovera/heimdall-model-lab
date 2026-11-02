"""Ledger chain hashing — the function that turns a LedgerRecord into the
`prev_hash` the next record must carry.

chain_hash(record) = canonical_sha256(record.hashable_dict())

`hashable_dict()` (defined on LedgerRecord itself) deliberately excludes
`body` (already committed via `body_hash`) and `signatures` (appended after
a record is chained; including them would be circular).
"""

from __future__ import annotations

from model_lab.canon.jcs import canonical_sha256
from model_lab.contracts.ledger_record import LedgerRecord


def chain_hash(record: LedgerRecord) -> str:
    return canonical_sha256(record.hashable_dict())


def compute_body_hash(body: dict) -> str:
    return canonical_sha256(body)
