"""AT-02: concurrent writers and torn-tail recovery lose no record.

**P0-3 scope note** (adversarial review, same root cause as AT-01 finding
C.3): the crash-repair machinery here (`repair_torn_tail`) and a malicious
tail-truncation are mechanically indistinguishable from the ledger file
alone -- both look like "the file ends earlier than some larger state might
have been." This file's tests only exercise the legitimate crash-repair
path; distinguishing accidental/benign tail-shortening from a deliberate
truncation is exactly what checkpoint-anchored completeness
(`ledger.anchor.verify_ledger_completeness`, exercised in
`test_at01_ledger_tamper.py`) is for, not this file.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.ledger.storage import LedgerFile, TornTailRepair, repair_torn_tail
from model_lab.ledger.verifier import verify_chain
from model_lab.ledger.writer import LedgerWriter

pytestmark = pytest.mark.acceptance


def test_concurrent_writers_serialize_without_lost_or_duplicate_seq(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    n_writers = 8
    n_each = 15

    def _write_batch(writer_id: int) -> None:
        w = LedgerWriter(path, author_id=f"writer-{writer_id}")
        for i in range(n_each):
            w.append(LedgerRecordType.HYPOTHESIS, {"writer": writer_id, "i": i})

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as pool:
        futures = [pool.submit(_write_batch, wid) for wid in range(n_writers)]
        for f in futures:
            f.result()

    records = LedgerFile(path).read_all()
    assert len(records) == n_writers * n_each

    seqs = [r.seq for r in records]
    assert seqs == list(range(len(records))), "seq must be a contiguous 0..N-1 run with no gaps or dupes"

    result = verify_chain(records)
    assert result.ok, f"chain must verify after concurrent writes: {result.violations}"

    # Every (writer, i) pair was actually written exactly once (no writer's
    # batch was silently lost).
    pairs = {(r.body["writer"], r.body["i"]) for r in records}
    assert pairs == {(w, i) for w in range(n_writers) for i in range(n_each)}


def test_torn_tail_is_detected_and_repaired_without_losing_prior_records(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    w = LedgerWriter(path, author_id="test-author")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "record 0"})
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "record 1"})

    good_bytes_before_tear = path.read_bytes()

    # Simulate a crash mid-write: append a truncated, non-JSON-parseable
    # partial line (no trailing newline) after the two good records.
    with open(path, "ab") as f:
        f.write(b'{"seq": 2, "record_type": "hypothesis", "body": {"text": "record 2 INCOMPL')

    # read_all with auto_repair=False must still surface the good records
    # and simply drop (not crash on) the torn tail.
    records_before_repair = LedgerFile(path).read_all(auto_repair=False)
    assert len(records_before_repair) == 2

    repair = repair_torn_tail(path)
    assert isinstance(repair, TornTailRepair)
    assert repair.dropped_at_line_index == 2

    after_bytes = path.read_bytes()
    assert after_bytes == good_bytes_before_tear

    records_after = LedgerFile(path).read_all()
    assert len(records_after) == 2
    result = verify_chain(records_after)
    assert result.ok

    # The ledger is writable again after repair — the next append continues
    # the sequence cleanly from seq=2.
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "record 2 (rewritten)"})
    records_final = LedgerFile(path).read_all()
    assert [r.seq for r in records_final] == [0, 1, 2]
    assert verify_chain(records_final).ok


def test_non_tail_corruption_is_never_silently_repaired(tmp_path: Path) -> None:
    from model_lab.ledger.storage import CorruptLedgerError

    path = tmp_path / "ledger.ndjson"
    w = LedgerWriter(path, author_id="test-author")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "record 0"})
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "record 1"})

    lines = path.read_bytes().split(b"\n")
    lines[0] = b"not json at all {{{"  # corrupt a NON-tail line
    path.write_bytes(b"\n".join(lines))

    with pytest.raises(CorruptLedgerError):
        LedgerFile(path).read_all()
