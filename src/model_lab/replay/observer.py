"""Observer-side InputBundle builder — §16, read-only against the AMC root.

Real schema confirmed by direct read-only inspection this session:
- `amc4_journal.ndjson`: NDJSON, 17 distinct `record_type` values, common
  envelope fields `record_type`, `record_ts_utc`, `record_fingerprint`,
  `entry_id`, `as_of_utc`, `experiment_id`, `cycle_id` (present on most but
  not all types). `amc4_source_health_snapshot` records carry
  `sources: [{provider_id, data_label, dependency_class, entitlement,
  error, freshness, last_success, latency_ms, status}]`.
- `cycles/cycle-{NNN}-{YYYYMMDD}_candidates.json`: top-level JSON array
  (often `[]`); `cycle-004-20260908` and a few others also have matching
  `_problem.json` / `_proposal.json` siblings.
- `cycle_id` format: `cycle-{NNN zero-padded}-{YYYYMMDD}`, numeric suffix
  restarts per experiment, date suffix is fixed to the experiment's
  inception date (not the cycle's actual run date).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.replay.amc_root import AMC_ROOT, CYCLES_DIR, JOURNAL_PATH, PINNED_EXPERIMENT_ID
from model_lab.replay.contracts import InputBundle
from model_lab.replay.journal_reader import (
    compute_prefix_hash,
    parse_ndjson_prefix,
    read_prefix_bytes,
)


def list_cycle_files(cycle_id: str) -> list[Path]:
    """Every file under cycles/ whose name starts with `{cycle_id}_`
    (candidates.json and, if present, problem.json/proposal.json), sorted
    for a deterministic hash list."""
    if not CYCLES_DIR.exists():
        return []
    return sorted(CYCLES_DIR.glob(f"{cycle_id}_*.json"))


def list_known_cycle_ids() -> list[str]:
    """Derive the distinct cycle_ids present on disk from the candidate
    filenames (read-only directory listing, no journal parsing needed)."""
    if not CYCLES_DIR.exists():
        return []
    ids: set[str] = set()
    for p in CYCLES_DIR.glob("cycle-*_candidates.json"):
        ids.add(p.name.removesuffix("_candidates.json"))
    return sorted(ids)


def _latest_source_health_snapshot_hash(
    journal_records: list[dict], experiment_id: str, cycle_id: str
) -> str | None:
    matches_cycle = [
        r
        for r in journal_records
        if r.get("record_type") == "amc4_source_health_snapshot"
        and r.get("experiment_id") == experiment_id
        and r.get("cycle_id") == cycle_id
    ]
    if matches_cycle:
        return sha256_bytes(_stable_bytes(matches_cycle[-1]))
    matches_experiment = [
        r
        for r in journal_records
        if r.get("record_type") == "amc4_source_health_snapshot" and r.get("experiment_id") == experiment_id
    ]
    if matches_experiment:
        return sha256_bytes(_stable_bytes(matches_experiment[-1]))
    return None


def _stable_bytes(obj: dict) -> bytes:
    import json

    return json.dumps(obj, sort_keys=True).encode("utf-8")


def build_input_bundle(cycle_id: str, *, experiment_id: str = PINNED_EXPERIMENT_ID) -> InputBundle:
    """Build one InputBundle for `cycle_id`, reading only what is on disk
    right now (read-only; no AMC CLI is ever invoked)."""
    journal_bytes = read_prefix_bytes(JOURNAL_PATH)
    byte_length, journal_hash = compute_prefix_hash(journal_bytes)
    journal_records = parse_ndjson_prefix(journal_bytes)

    candidate_hashes = [sha256_bytes(p.read_bytes()) for p in list_cycle_files(cycle_id)]
    health_hash = _latest_source_health_snapshot_hash(journal_records, experiment_id, cycle_id)

    return InputBundle(
        cycle_id=cycle_id,
        journal_prefix_hash=journal_hash,
        journal_prefix_byte_length=byte_length,
        candidate_file_hashes=candidate_hashes,
        source_health_snapshot_hash=health_hash,
        observed_at_utc=datetime.now(UTC),
        root_path=str(AMC_ROOT),
        experiment_id=experiment_id,
    )
