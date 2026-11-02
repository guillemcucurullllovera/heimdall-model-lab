"""Append-only NDJSON storage for a single ledger file: `flock` + `fsync` +
torn-tail detection/repair, per MODEL1_foundation_contracts.md §12.

One ledger file = one totally-ordered sequence of LedgerRecords (seq 0..N).
MODEL-1 may run several named ledgers (e.g. one per purpose/namespace); each
is an independent instance of `LedgerFile`.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from model_lab.contracts.ledger_record import LedgerRecord


class CorruptLedgerError(Exception):
    """A non-tail line failed to parse. This is NOT the expected torn-tail
    crash scenario (which only ever affects the last line) — it indicates
    either tampering or a deeper corruption, and is never silently
    repaired."""


@dataclass(frozen=True)
class TornTailRepair:
    dropped_bytes: bytes
    dropped_at_line_index: int


def _split_lines(data: bytes) -> list[bytes]:
    if not data:
        return []
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def read_raw_lines_with_torn_tail_check(path: Path) -> tuple[list[bytes], TornTailRepair | None]:
    """Read all lines; if the *last* line fails to parse as JSON, treat it
    as a torn tail (a crash mid-`write()`) and report it separately rather
    than raising. Any non-tail line that fails to parse raises
    CorruptLedgerError — that is tamper/corruption, not a torn write, and
    must never be silently dropped."""
    if not path.exists():
        return [], None
    data = path.read_bytes()
    lines = _split_lines(data)
    if not lines:
        return [], None
    good: list[bytes] = []
    torn: TornTailRepair | None = None
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                torn = TornTailRepair(dropped_bytes=line, dropped_at_line_index=i)
                break
            raise CorruptLedgerError(
                f"{path}: line {i} (of {len(lines)}) is not valid JSON and is not the "
                f"file's last line — this is corruption/tampering, not a torn write, "
                f"and cannot be auto-repaired: {exc}"
            ) from exc
        good.append(line)
    return good, torn


def repair_torn_tail(path: Path) -> TornTailRepair | None:
    """Physically truncate a torn last line from `path`, if present, under
    an exclusive lock. Returns the repair record (for the caller to log as
    an `incident` ledger record) or None if nothing needed repair."""
    with _locked(path):
        good, torn = read_raw_lines_with_torn_tail_check(path)
        if torn is None:
            return None
        with open(path, "wb") as f:
            for line in good:
                f.write(line)
                f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
        return torn


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_all_unlocked(path: Path, *, auto_repair: bool = False) -> list[LedgerRecord]:
    """Core read, assuming the caller already holds the file lock (or does
    not need it, e.g. a read-only verifier run against a quiescent ledger).
    Never call this without either holding `_locked(path)` yourself or
    accepting the (rare, benign for read-only verification) race against a
    concurrent writer."""
    good, torn = read_raw_lines_with_torn_tail_check(path)
    if torn is not None and auto_repair:
        with open(path, "wb") as f:
            for line in good:
                f.write(line)
                f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    return [LedgerRecord.model_validate_json(line) for line in good]


class LedgerFile:
    """One append-only NDJSON ledger. Concurrency-safe via `flock` on a
    sibling `.lock` file (held for the full read-last-record + append
    critical section, so concurrent writers serialize rather than race —
    AT-02)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read_all(self, *, auto_repair: bool = False) -> list[LedgerRecord]:
        with _locked(self.path):
            return _read_all_unlocked(self.path, auto_repair=auto_repair)

    def last_record(self, *, auto_repair: bool = False) -> LedgerRecord | None:
        records = self.read_all(auto_repair=auto_repair)
        return records[-1] if records else None

    def append(self, record: LedgerRecord) -> LedgerRecord:
        """Append a single, already-fully-formed record under an exclusive
        lock, with fsync before releasing the lock (durability: a reader
        that observes the lock released always observes the write)."""
        with _locked(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = record.model_dump_json()
            with open(self.path, "ab") as f:
                f.write(line.encode("utf-8"))
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())
        return record
