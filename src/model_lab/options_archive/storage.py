"""Minimal append-only, hash-chained NDJSON storage for the option archive.

Reuses the same `flock`+`fsync` locking primitive as the evidence ledger
(`model_lab.ledger.storage._locked`) but is intentionally simpler: the
option archive does not need governance/signing machinery, only §18a rule 1
("append-only with prev_hash chaining and periodic operator-signed
checkpoints" — the checkpoint-signing part is out of scope here, same as
the evidence ledger's operator checkpoints).
"""

from __future__ import annotations

import os
from pathlib import Path

from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.contracts.option_snapshot import OptionSnapshotRecord
from model_lab.ledger.storage import _locked


def _read_all_unlocked(path: Path) -> list[OptionSnapshotRecord]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(OptionSnapshotRecord.model_validate_json(line))
    return records


class OptionArchiveFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_all(self) -> list[OptionSnapshotRecord]:
        with _locked(self.path):
            return _read_all_unlocked(self.path)

    def append_many(self, build_records) -> list[OptionSnapshotRecord]:
        """`build_records(existing: list[OptionSnapshotRecord]) -> list[OptionSnapshotRecord]`
        — called once, under the lock, with the current archive contents, and
        must return the FULL list of new records to append (already carrying
        correct archive_seq/prev_hash/record_hash, computed by the caller
        using `next_seq_and_prev_hash` below). Kept as a callback so a whole
        batch (e.g. one ingested snapshot file) is appended atomically."""
        with _locked(self.path):
            existing = _read_all_unlocked(self.path)
            new_records = build_records(existing)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for rec in new_records:
                    f.write(rec.model_dump_json())
                    f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            return new_records


def record_hash_of(record: OptionSnapshotRecord) -> str:
    """The record's own hash, computed the same self-referential way as the
    gate v2 spec's own convention: canonical hash of the record excluding
    its own record_hash field."""
    return canonical_sha256_excluding(record.canonical_dict(), "record_hash")


def next_seq_and_prev_hash(existing: list[OptionSnapshotRecord]) -> tuple[int, str | None]:
    if not existing:
        return 0, None
    last = existing[-1]
    return last.archive_seq + 1, last.record_hash
