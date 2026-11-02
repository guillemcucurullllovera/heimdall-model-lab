"""DatasetSnapshot builder: content-addressed raw payload storage, PIT
long-table writer, gap/DQ detection, first-bar assertion — MODEL1_foundation
_contracts.md §9.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas_market_calendars as mcal

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    KnownGap,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, derive_research_state

DATA_ESTATE_ROOT = Path("./local-data")
RAW_ROOT = DATA_ESTATE_ROOT / "raw"
SNAPSHOTS_ROOT = DATA_ESTATE_ROOT / "snapshots"


def store_raw(data: bytes) -> str:
    """Content-addressed raw payload store: raw/<sha256[:2]>/<sha256>. Files
    are made read-only after write (best-effort chmod; not relied on for
    security, only as a footgun guard against accidental in-place edits)."""
    digest = sha256_bytes(data)
    subdir = RAW_ROOT / digest[:2]
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / digest
    if not path.exists():
        path.write_bytes(data)
        path.chmod(0o444)
    return digest


def write_rows_ndjson(snapshot_id: str, rows: Iterable[PitRow]) -> tuple[str, str]:
    """Writes the PIT long table to snapshots/<snapshot_id>/rows.ndjson.
    Returns (relative_path, logical_content_hash)."""
    snap_dir = SNAPSHOTS_ROOT / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / "rows.ndjson"
    with open(out_path, "wb") as f:
        for row in rows:
            f.write(row.model_dump_json().encode("utf-8"))
            f.write(b"\n")
    content_hash = sha256_bytes(out_path.read_bytes())
    return str(out_path.relative_to(DATA_ESTATE_ROOT)), content_hash


def load_rows_ndjson(rows_path: str) -> list[PitRow]:
    """Read-only sibling of `write_rows_ndjson`: loads a snapshot's row
    content back from disk given the relative `rows_path` recorded on its
    `DatasetSnapshot` (relative to `DATA_ESTATE_ROOT`).

    Needed by `DatasetSnapshotRegistry.get_rows` (MODEL-2A P1 remediation,
    `MODEL1_REMEDIATION_REREVIEW.md` §I): a companion corporate-actions
    snapshot's "zero events" claim can only be mechanically verified by
    reading its actual row content, not by confirming the snapshot merely
    exists."""
    full_path = DATA_ESTATE_ROOT / rows_path
    rows: list[PitRow] = []
    with open(full_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(PitRow.model_validate_json(line))
    return rows


@dataclass(frozen=True)
class DqReport:
    n_rows: int
    n_duplicate_dates: int
    n_non_monotonic: int
    n_gap_days: int
    known_gaps: list[KnownGap]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_duplicate_dates": self.n_duplicate_dates,
            "n_non_monotonic": self.n_non_monotonic,
            "n_gap_days": self.n_gap_days,
            "known_gaps": [g.canonical_dict() for g in self.known_gaps],
        }


_XNYS_CALENDAR = mcal.get_calendar("XNYS")


def compute_calendar_gaps(obs_dates: list[date], *, exchange: str = "XNYS") -> list[KnownGap]:
    """Gap detection against the REAL exchange trading calendar (via
    `pandas_market_calendars`), not a weekday approximation.

    Remediation note (MODEL1_ADVERSARIAL_REVIEW.md finding E.2): the prior
    weekday-approximation method both mis-arithmeticked its own claim
    (reported ~138 when even its own formula gives 102) and was
    conceptually wrong regardless (it flags ordinary exchange holidays as
    'gaps' that aren't gaps at all). Independently re-verified for this fix:
    the true XNYS-calendar gap for all 6 ETFs over their full acquired
    windows is 0 (every expected trading day is present, no unexpected
    extra rows) — this function computes that comparison directly rather
    than approximating it."""
    if not obs_dates:
        return []
    sorted_dates = sorted(obs_dates)
    calendar = mcal.get_calendar(exchange) if exchange != "XNYS" else _XNYS_CALENDAR
    schedule = calendar.schedule(start_date=sorted_dates[0], end_date=sorted_dates[-1])
    expected = {d.date() if hasattr(d, "date") else d for d in schedule.index}
    have = set(sorted_dates)
    missing = sorted(expected - have)
    if not missing:
        return []
    # Collapse consecutive missing trading days into contiguous KnownGap ranges.
    gaps: list[KnownGap] = []
    run_start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days <= 3:  # tolerate short internal calendar breaks within one contiguous gap
            prev = d
            continue
        gaps.append(KnownGap(start=run_start, end=prev, reason="missing expected XNYS trading day(s)"))
        run_start = d
        prev = d
    gaps.append(KnownGap(start=run_start, end=prev, reason="missing expected XNYS trading day(s)"))
    return gaps


def build_dq_report(rows: list[PitRow], *, exchange: str = "XNYS") -> DqReport:
    dates = [r.observation_period for r in rows]
    seen: set[date] = set()
    dup = 0
    for d in dates:
        if d in seen:
            dup += 1
        seen.add(d)
    non_monotonic = sum(1 for a, b in zip(dates, dates[1:], strict=False) if b < a)
    gaps = compute_calendar_gaps(dates, exchange=exchange)
    gap_days = sum((g.end - g.start).days + 1 for g in gaps)
    return DqReport(
        n_rows=len(rows),
        n_duplicate_dates=dup,
        n_non_monotonic=non_monotonic,
        n_gap_days=gap_days,
        known_gaps=gaps,
    )


def write_dq_report(snapshot_id: str, dq: DqReport) -> str:
    snap_dir = SNAPSHOTS_ROOT / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / "dq_report.json"
    out_path.write_text(json.dumps(dq.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_bytes(out_path.read_bytes())


def build_etf_bar_snapshot(
    *,
    snapshot_id: str,
    source_id: str,
    symbol: str,
    endpoint_or_path: str,
    params: dict[str, str],
    licence_status: LicenceStatus,
    rows: list[PitRow],
    raw_payload_hashes: list[str],
    requested_start: date | None,
    adjustment_kind: str,
    retrieved_at_utc: datetime | None = None,
    adjustment_method: str | None = None,
    corporate_actions_snapshot_id: str | None = None,
) -> DatasetSnapshot:
    """Builds a DatasetSnapshot for one ETF's daily-bar history from one
    provider. Writes rows.ndjson + dq_report.json to the data estate as a
    side effect. `adjustment_method`/`corporate_actions_snapshot_id` name
    the transformation's own lineage for a DERIVED (adjusted) snapshot —
    both None for a raw, unadjusted snapshot."""
    rows_path, logical_content_hash = write_rows_ndjson(snapshot_id, rows)
    dq = build_dq_report(rows)
    dq_hash = write_dq_report(snapshot_id, dq)

    obs_dates = [r.observation_period for r in rows]
    first_bar_date = min(obs_dates) if obs_dates else None
    last_bar_date = max(obs_dates) if obs_dates else None
    first_bar_lag_days = None
    if requested_start is not None and first_bar_date is not None and first_bar_date != requested_start:
        first_bar_lag_days = (first_bar_date - requested_start).days

    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        source_id=source_id,
        endpoint_or_path=endpoint_or_path,
        params=params,
        retrieved_at_utc=retrieved_at_utc or datetime.now(UTC),
        licence_status=licence_status,
        research_state=derive_research_state(licence_status),
        raw_payload_hashes=raw_payload_hashes,
        logical_content_hash=logical_content_hash,
        schema_id="ohlcv_daily.v1",
        rows=None,
        rows_path=rows_path,
        calendar=CalendarInfo(exchange="XNYS", holidays_version=None, timezone="America/New_York"),
        adjustments=AdjustmentInfo(
            kind=adjustment_kind, method=adjustment_method, corporate_actions_snapshot_id=corporate_actions_snapshot_id
        ),
        dq_report_hash=dq_hash,
        known_gaps=dq.known_gaps,
        pit_class=PitClass.AS_RELEASED,
        requested_start=requested_start,
        first_bar_date=first_bar_date,
        first_bar_lag_days=first_bar_lag_days,
        last_bar_date=last_bar_date,
    )


def build_reference_snapshot(
    *,
    snapshot_id: str,
    source_id: str,
    endpoint_or_path: str,
    params: dict[str, str],
    licence_status: LicenceStatus,
    rows: list[PitRow],
    raw_payload_hashes: list[str],
    schema_id: str,
    retrieved_at_utc: datetime | None = None,
) -> DatasetSnapshot:
    """Builds a DatasetSnapshot for reference/event metadata (e.g. corporate
    actions) that is not a daily-bar time series — no calendar-gap DQ check
    applies (a splits table has no expected trading-day cadence), but the
    same content-addressed raw storage, PIT row writing, and licence/
    research_state mechanics apply identically. `pit_class=SNAPSHOT_ONLY`:
    a reference table fetched once, at `retrieved_at_utc`, not a per-row
    vintage history (each row still carries its own `knowledge_time_utc`,
    the event's own execution/announcement date where known)."""
    rows_path, logical_content_hash = write_rows_ndjson(snapshot_id, rows)
    dq = DqReport(n_rows=len(rows), n_duplicate_dates=0, n_non_monotonic=0, n_gap_days=0, known_gaps=[])
    dq_hash = write_dq_report(snapshot_id, dq)

    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        source_id=source_id,
        endpoint_or_path=endpoint_or_path,
        params=params,
        retrieved_at_utc=retrieved_at_utc or datetime.now(UTC),
        licence_status=licence_status,
        research_state=derive_research_state(licence_status),
        raw_payload_hashes=raw_payload_hashes,
        logical_content_hash=logical_content_hash,
        schema_id=schema_id,
        rows=None,
        rows_path=rows_path,
        calendar=CalendarInfo(exchange=None, holidays_version=None, timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="none"),
        dq_report_hash=dq_hash,
        known_gaps=[],
        pit_class=PitClass.SNAPSHOT_ONLY,
        requested_start=None,
        first_bar_date=None,
        first_bar_lag_days=None,
        last_bar_date=None,
    )
