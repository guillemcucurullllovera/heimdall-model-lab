"""High-level ledger writer: computes seq/prev_hash/body_hash and appends,
under the same lock used for reading the current tail (so two concurrent
writers can never both compute the same next seq — AT-02).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_lab.canon.jcs import canonical_sha256
from model_lab.contracts.ledger_record import LedgerRecord, LedgerRecordType, Signature
from model_lab.ledger.chain import chain_hash
from model_lab.ledger.storage import LedgerFile, _locked, _read_all_unlocked

PreAppendCheck = Callable[[LedgerRecord, list[LedgerRecord]], None]


class LedgerWriter:
    def __init__(self, path: Path, author_id: str) -> None:
        self._file = LedgerFile(path)
        self.author_id = author_id

    def append(
        self,
        record_type: LedgerRecordType,
        body: dict[str, Any],
        *,
        signatures: list[Signature] | None = None,
        pre_append_check: PreAppendCheck | None = None,
    ) -> LedgerRecord:
        """Build and append a new record. `pre_append_check`, if given, is
        called with (candidate_record, existing_records) while still holding
        the write lock, and may raise to reject the append (e.g. the
        BLOCKED-task-prereg rule in governance.py) — this makes the check
        atomic with the append, closing the TOCTOU window a caller-side
        check would leave open.
        """
        with _locked(self._file.path):
            # NOTE: must use the unlocked core read here, not
            # self._file.read_all() — that re-acquires the same lock and
            # would deadlock against the outer `with _locked(...)` above
            # (flock is per-open-file-description, not reentrant within a
            # process across separate os.open() calls on the same path).
            existing = _read_all_unlocked(self._file.path)
            last = existing[-1] if existing else None
            seq = (last.seq + 1) if last else 0
            prev_hash = chain_hash(last) if last else None
            record = LedgerRecord(
                seq=seq,
                prev_hash=prev_hash,
                record_type=record_type,
                body=body,
                body_hash=canonical_sha256(body),
                author_id=self.author_id,
                ts_utc=datetime.now(UTC),
                signatures=signatures or [],
            )
            if pre_append_check is not None:
                pre_append_check(record, existing)
            line = record.model_dump_json()
            self._file.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file.path, "ab") as f:
                f.write(line.encode("utf-8"))
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())
            return record
