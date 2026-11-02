"""11-field MacroStateVector PIT panel builder — own process, own rate cap,
own cache under ./local-data/macro_pit/. This is a
completely independent implementation; it never imports from or writes to
heimdall-command-center.

Field -> FRED series ID mapping (confirmed by prior research against
heimdall-command-center's `research/macro/state_vector.py`, read-only):
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, derive_research_state
from model_lab.data.providers.fred_alfred import (
    ChunkedVintageFetchResult,
    FredClient,
    VintageFetchResult,
)

MACRO_PIT_ROOT = Path("./local-data/macro_pit")

# Series whose full (observation x vintage) cross-product exceeds
# fetch_all_vintages's single-request practical row cap (NFCI:
# 925,555 rows -- see MODEL1_ADVERSARIAL_REVIEW.md finding D.3/Q.1) and must
# use the chunked, observation-year-partitioned fetch instead, so the most
# recent observations are never silently dropped.
SERIES_REQUIRING_CHUNKED_FETCH: frozenset[str] = frozenset({"NFCI"})

MACRO_STATE_VECTOR_FIELDS: dict[str, str] = {
    "growth": "INDPRO",
    "inflation_headline": "CPIAUCSL",
    "inflation_breakeven": "T10YIE",
    "labor_unemployment": "UNRATE",
    "labor_sahm": "SAHMREALTIME",
    "policy_rate": "DFF",
    "yield_3m": "DGS3MO",
    "yield_2y": "DGS2",
    "yield_10y": "DGS10",
    "financial_conditions": "NFCI",
    "liquidity_m2": "M2SL",
}


@dataclass(frozen=True)
class MacroFieldCoverage:
    field: str
    series_id: str
    frequency: str
    n_observations: int
    earliest_pit_date: str | None
    latest_pit_date: str | None
    n_vintages: int
    vintage_query_degraded: bool
    degraded_reason: str | None
    total_rows_available: int | None
    truncated: bool
    usable: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "series": self.series_id,
            "field": self.field,
            "frequency": self.frequency,
            "n_observations": self.n_observations,
            "earliest_pit_date": self.earliest_pit_date,
            "latest_pit_date": self.latest_pit_date,
            "revision_depth_n_vintages": self.n_vintages,
            "vintage_query_degraded": self.vintage_query_degraded,
            "degraded_reason": self.degraded_reason,
            "total_rows_available_at_source": self.total_rows_available,
            "truncated_at_fetch_cap": self.truncated,
            "usable": self.usable,
            "reason": self.reason,
        }


def _rows_to_pit_rows(result: VintageFetchResult | ChunkedVintageFetchResult, raw_hash: str) -> list[PitRow]:
    rows: list[PitRow] = []
    for obs in result.observations:
        if obs.value is None:
            continue
        rows.append(
            PitRow(
                series_id=obs.series_id,
                observation_period=obs.observation_period,
                value=obs.value,
                realtime_start=obs.realtime_start,
                realtime_end=obs.realtime_end,
                knowledge_time_utc=datetime.combine(
                    obs.realtime_start, datetime.min.time(), tzinfo=UTC
                ),
                source_snapshot_sha=raw_hash,
            )
        )
    return rows


def build_macro_pit_panel() -> tuple[list[DatasetSnapshot], list[MacroFieldCoverage]]:
    """Fetches all 11 series' full ALFRED vintage history, writes raw
    payloads content-addressed and per-series PIT row tables under
    macro_pit/, and returns (snapshots, coverage)."""
    MACRO_PIT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshots: list[DatasetSnapshot] = []
    coverage: list[MacroFieldCoverage] = []

    with FredClient() as client:
        for field, series_id in MACRO_STATE_VECTOR_FIELDS.items():
            series_dir = MACRO_PIT_ROOT / series_id
            series_dir.mkdir(parents=True, exist_ok=True)

            if series_id in SERIES_REQUIRING_CHUNKED_FETCH:
                chunked = client.fetch_all_vintages_full_history(
                    series_id, series_start=date(1971, 1, 1), series_end=date.today(), chunk_years=5
                )
                raw_hashes = []
                for i, payload in enumerate(chunked.raw_payloads):
                    chunk = chunked.chunks[i]
                    chunk_path = series_dir / f"raw_chunk_{chunk['observation_start']}_{chunk['observation_end']}.json"
                    chunk_path.write_bytes(payload)
                    raw_hashes.append(sha256_bytes(payload))
                combined_hash = sha256_bytes(b"".join(chunked.raw_payloads))
                pit_rows = _rows_to_pit_rows(chunked, combined_hash)
                degraded: bool = chunked.any_chunk_degraded
                degraded_reason: str | None = (
                    "; ".join(str(c.get("degraded_reason", "")) for c in chunked.chunks if c.get("vintage_query_degraded")) or None
                )
                truncated: bool = chunked.any_chunk_truncated
                total_rows_available: int | None = chunked.total_rows
                earliest_obs, latest_obs, n_vintages = (
                    chunked.earliest_observation_period,
                    chunked.latest_observation_period,
                    chunked.n_distinct_realtime_starts,
                )
                raw_payload_hashes_for_snapshot = raw_hashes
            else:
                result = client.fetch_all_vintages(series_id)
                raw_path = series_dir / "raw_response.json"
                raw_path.write_bytes(result.raw_payload)
                raw_hash = sha256_bytes(result.raw_payload)
                pit_rows = _rows_to_pit_rows(result, raw_hash)
                degraded, degraded_reason, truncated, total_rows_available = (
                    result.vintage_query_degraded,
                    result.degraded_reason,
                    result.truncated,
                    result.total_rows_available,
                )
                earliest_obs, latest_obs, n_vintages = (
                    result.earliest_observation_period,
                    result.latest_observation_period,
                    result.n_distinct_realtime_starts,
                )
                raw_payload_hashes_for_snapshot = [raw_hash]

            rows_path = series_dir / "rows.ndjson"
            with open(rows_path, "wb") as f:
                for row in pit_rows:
                    f.write(row.model_dump_json().encode("utf-8"))
                    f.write(b"\n")
            logical_hash = sha256_bytes(rows_path.read_bytes())

            usable = len(pit_rows) > 0
            reason = "fetched successfully" if usable else "no observations returned"
            if degraded:
                reason = degraded_reason or "vintage query degraded"
            if truncated:
                reason += f" (TRUNCATED at fetch cap: {len(pit_rows)}/{total_rows_available} rows)"

            frequency = "unknown"
            if series_id in ("DFF", "DGS3MO", "DGS2", "DGS10", "T10YIE", "NFCI"):
                frequency = "daily" if series_id != "NFCI" else "weekly"
            elif series_id in ("INDPRO", "CPIAUCSL", "UNRATE", "SAHMREALTIME", "M2SL"):
                frequency = "monthly"

            coverage.append(
                MacroFieldCoverage(
                    field=field,
                    series_id=series_id,
                    frequency=frequency,
                    n_observations=len(pit_rows),
                    earliest_pit_date=str(earliest_obs) if earliest_obs else None,
                    latest_pit_date=str(latest_obs) if latest_obs else None,
                    n_vintages=n_vintages,
                    vintage_query_degraded=degraded,
                    degraded_reason=degraded_reason,
                    total_rows_available=total_rows_available,
                    truncated=truncated,
                    usable=usable,
                    reason=reason,
                )
            )

            snapshots.append(
                DatasetSnapshot(
                    snapshot_id=f"macro_pit_{series_id}",
                    source_id="fred_alfred",
                    endpoint_or_path="/fred/series/observations",
                    params={"series_id": series_id, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"},
                    retrieved_at_utc=datetime.now(UTC),
                    licence_status=LicenceStatus.CLEAR,
                    research_state=derive_research_state(LicenceStatus.CLEAR),
                    raw_payload_hashes=raw_payload_hashes_for_snapshot,
                    logical_content_hash=logical_hash,
                    schema_id="macro_pit_vintage.v1",
                    rows=None,
                    rows_path=str(rows_path.relative_to(MACRO_PIT_ROOT.parent)),
                    calendar=CalendarInfo(exchange=None, holidays_version=None, timezone="America/New_York"),
                    adjustments=AdjustmentInfo(kind="none"),
                    known_gaps=[],
                    pit_class=PitClass.AS_RELEASED if not degraded else PitClass.SNAPSHOT_ONLY,
                )
            )

    return snapshots, coverage


def write_coverage_reports(coverage: list[MacroFieldCoverage], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "macro_pit_coverage.json"
    json_path.write_text(
        json.dumps({"fields": [c.to_dict() for c in coverage]}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md_path = out_dir / "MACRO_PIT_COVERAGE.md"
    lines = [
        "# Macro PIT Coverage",
        "",
        "11-field MacroStateVector PIT panel, built independently by MODEL-1 from FRED/ALFRED "
        "(own process, own rate cap, own cache; heimdall-command-center's macro files/database/"
        "worktree were never read or modified).",
        "",
        "| Field | Series | Frequency | n_obs | Earliest PIT | Latest PIT | Revision depth (vintages) | Usable | Reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in coverage:
        lines.append(
            f"| {c.field} | {c.series_id} | {c.frequency} | {c.n_observations} | "
            f"{c.earliest_pit_date} | {c.latest_pit_date} | {c.n_vintages} | "
            f"{'YES' if c.usable else 'NO'} | {c.reason} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
