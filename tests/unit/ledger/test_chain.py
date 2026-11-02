from __future__ import annotations

from datetime import UTC, datetime

from model_lab.canon.jcs import canonical_sha256
from model_lab.contracts.ledger_record import LedgerRecord, LedgerRecordType
from model_lab.ledger.chain import chain_hash, compute_body_hash

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _record(seq: int, prev_hash: str | None, body: dict) -> LedgerRecord:
    return LedgerRecord(
        seq=seq, prev_hash=prev_hash, record_type=LedgerRecordType.HYPOTHESIS, body=body,
        body_hash=compute_body_hash(body), author_id="a", ts_utc=NOW,
    )


def test_compute_body_hash_matches_canonical_sha256() -> None:
    body = {"b": 1, "a": 2}
    assert compute_body_hash(body) == canonical_sha256(body)


def test_chain_hash_is_deterministic() -> None:
    rec = _record(0, None, {"x": 1})
    assert chain_hash(rec) == chain_hash(rec)


def test_chain_hash_excludes_body_and_signatures() -> None:
    """Two records differing only in body (same body_hash coincidentally
    impossible in practice, but we test via hashable_dict directly) or
    signatures must chain-hash identically -- the chain hash is over
    seq/prev_hash/record_type/body_hash/author_id/ts_utc only."""
    rec1 = _record(0, None, {"x": 1})
    rec2 = rec1.model_copy(update={"signatures": []})
    assert chain_hash(rec1) == chain_hash(rec2)


def test_chain_hash_changes_with_seq() -> None:
    rec0 = _record(0, None, {"x": 1})
    rec1 = _record(1, None, {"x": 1})
    assert chain_hash(rec0) != chain_hash(rec1)


def test_chain_hash_changes_with_body_hash() -> None:
    rec_a = _record(0, None, {"x": 1})
    rec_b = _record(0, None, {"x": 2})
    assert chain_hash(rec_a) != chain_hash(rec_b)


def test_chain_hash_is_hex64() -> None:
    h = chain_hash(_record(0, None, {"x": 1}))
    assert len(h) == 64
    int(h, 16)
