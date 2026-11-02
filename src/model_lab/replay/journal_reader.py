"""Read-only journal observer utilities.

The AMC journal (`amc4_journal.ndjson`) is written by PAPER_AUTOPILOT
roughly every 6 hours and can grow while a MODEL-1 observation is in
progress. §16: "at each observation record the journal byte length and a
prefix hash, and read only up to that length." Unlike MODEL-1's own ledger
(ledger/storage.py), this file is NOT something MODEL-1 asserts integrity
over or repairs — a trailing partial line within the observed prefix is
simply not-yet-complete-when-we-looked, not corruption, so it is dropped
silently from the parsed record list (but still counted in the byte length/
hash, since the prefix hash must cover exactly the bytes actually read).
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes


def read_prefix_bytes(path: Path) -> bytes:
    """Read the file's current full contents. A snapshot at one instant —
    the caller should treat the returned length as *the* observed prefix
    length for this observation (do not re-read the file later and expect
    the same bytes back)."""
    return path.read_bytes()


def compute_prefix_hash(data: bytes) -> tuple[int, str]:
    """Returns (byte_length, sha256_hex) of the given prefix bytes."""
    return len(data), sha256_bytes(data)


def parse_ndjson_prefix(data: bytes) -> list[dict]:
    """Parse complete NDJSON lines from `data`; a trailing incomplete line
    (the file was mid-append when read) is dropped, not raised."""
    records: list[dict] = []
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # torn tail from a concurrent append; not an error here
            raise  # a non-tail line failing to parse is unexpected; surface it
    return records
