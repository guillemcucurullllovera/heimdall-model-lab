"""DatasetSnapshot/1 — MODEL1_foundation_contracts.md §9.

```
DatasetSnapshot/1 {
  snapshot_id, source_id, endpoint_or_path, params, retrieved_at_utc, licence_status,
  raw_payload_hashes[], logical_content_hash, schema_id,
  rows: long table (series_id, observation_period, value, realtime_start, realtime_end,
                     knowledge_time_utc, source_snapshot_sha),
  calendar: {exchange, holidays_version, timezone}, adjustments: {kind, method, corporate_actions_snapshot_id},
  dq_report_hash, datasheet_hash, known_gaps[], pit_class: AS_RELEASED | AS_REVISED | SNAPSHOT_ONLY
}
```

Rules (verbatim numbering from §9):
1. timestamp semantics are explicit per source (event time, vendor publication
   time, our conservative `available_at`, decision time); daily bars usable
   only after close + latency; embargo >= max(label horizon, publication lag).
2. first-release vintage for anything knowable in real time; latest-revised
   only for outcome labels defined as revised.
3. raw payloads stored unmodified and hashed; normalisation is a deterministic
   function recorded in the snapshot.
4. a snapshot with licence_status in {UNRESOLVED, RESTRICTED} is `quarantined`
   and can feed only `research_only_unlicensed` experiments.
5. AS_REVISED series are flagged and excluded from any PIT-required task.
6. `first_bar_date == requested_start` (or a declared lag) is asserted for
   every bar download.
7. a leakage harness runs on every FeatureSet (implemented in eval/, not here).

Storage note: for large snapshots (e.g. years of daily bars), the row-level
long table lives on disk as CSV/NDJSON under the data estate
(`heimdall-model-lab-data/snapshots/<snapshot_id>/rows.ndjson`), referenced
here by `rows_path` + `logical_content_hash` (hash of that file's bytes).
Small snapshots (e.g. a single macro series vintage pull) may embed `rows`
directly. Exactly one of `rows` / `rows_path` must be set.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.licence import LicenceStatus, ResearchState


class PitClass(StrEnum):
    AS_RELEASED = "AS_RELEASED"
    AS_REVISED = "AS_REVISED"
    SNAPSHOT_ONLY = "SNAPSHOT_ONLY"


class PitRow(ModelLabBaseModel):
    """One row of the PIT long table.

    **P1-4 remediation** (adversarial review finding D.4): "no
    `model_validator` anywhere enforces `knowledge_time_utc >=
    realtime_start` on `PitRow` -- a deliberately-backdated `PitRow`
    (knowledge_time_utc six years before realtime_start) is accepted
    silently by pydantic." Fixed below: a row cannot claim to have been
    knowable before its own vintage began, and `realtime_end` (when set)
    cannot precede `realtime_start`.
    """

    series_id: str
    observation_period: date
    value: str = Field(description="stored as a string to avoid float-canonicalization ambiguity")
    realtime_start: date | None = Field(default=None, description="vintage window start; None if unrevised/unknown")
    realtime_end: date | None = Field(default=None, description="vintage window end; None if still current")
    knowledge_time_utc: datetime = Field(
        description="our conservative available_at; when this observation was actually knowable"
    )
    source_snapshot_sha: Sha256Hex

    @model_validator(mode="after")
    def _check_temporal_ordering(self) -> PitRow:
        if self.realtime_start is not None and self.knowledge_time_utc.date() < self.realtime_start:
            raise ValueError(
                f"knowledge_time_utc ({self.knowledge_time_utc.date()}) is before realtime_start "
                f"({self.realtime_start}) -- a row cannot have been knowable before its own vintage "
                "began (P1-4 remediation, adversarial review finding D.4)"
            )
        if self.realtime_start is not None and self.realtime_end is not None and self.realtime_end < self.realtime_start:
            raise ValueError(
                f"realtime_end ({self.realtime_end}) is before realtime_start ({self.realtime_start})"
            )
        return self


class CalendarInfo(ModelLabBaseModel):
    exchange: str | None = None
    holidays_version: str | None = None
    timezone: str


class AdjustmentInfo(ModelLabBaseModel):
    kind: str = Field(description="e.g. 'none', 'split_dividend_adjusted', 'total_return'")
    method: str | None = None
    corporate_actions_snapshot_id: str | None = None


class KnownGap(ModelLabBaseModel):
    start: date
    end: date
    reason: str


class DatasetSnapshot(ModelLabBaseModel):
    schema_name: str = "DatasetSnapshot"
    schema_version: int = 1

    snapshot_id: str
    source_id: str
    endpoint_or_path: str
    params: dict[str, Any] = Field(default_factory=dict)
    retrieved_at_utc: datetime
    licence_status: LicenceStatus
    research_state: ResearchState = Field(
        description="must equal derive_research_state(licence_status); validated below"
    )

    raw_payload_hashes: list[Sha256Hex] = Field(default_factory=list)
    logical_content_hash: Sha256Hex
    schema_id: str

    rows: list[PitRow] | None = None
    rows_path: str | None = None

    calendar: CalendarInfo
    adjustments: AdjustmentInfo

    dq_report_hash: Sha256Hex | None = None
    datasheet_hash: Sha256Hex | None = None
    known_gaps: list[KnownGap] = Field(default_factory=list)
    pit_class: PitClass

    requested_start: date | None = None
    first_bar_date: date | None = None
    first_bar_lag_days: int | None = Field(
        default=None, description="declared lag if first_bar_date != requested_start (rule 6)"
    )
    last_bar_date: date | None = None

    @model_validator(mode="after")
    def _check_research_state_matches_licence(self) -> DatasetSnapshot:
        from model_lab.contracts.licence import derive_research_state

        expected = derive_research_state(LicenceStatus(self.licence_status))
        if ResearchState(self.research_state) != expected:
            raise ValueError(
                f"research_state {self.research_state!r} does not match the mechanical "
                f"derivation from licence_status {self.licence_status!r} "
                f"(expected {expected!r}). research_state must never be hand-assigned."
            )
        return self

    @model_validator(mode="after")
    def _check_exactly_one_row_source(self) -> DatasetSnapshot:
        if (self.rows is None) == (self.rows_path is None):
            raise ValueError("exactly one of rows / rows_path must be set")
        return self

    @model_validator(mode="after")
    def _check_first_bar_assertion(self) -> DatasetSnapshot:
        if (
            self.requested_start is not None
            and self.first_bar_date is not None
            and self.first_bar_date != self.requested_start
            and self.first_bar_lag_days is None
        ):
            raise ValueError(
                "first_bar_date != requested_start but no first_bar_lag_days was declared "
                "(rule 6: the lag must be explicit, not silent)"
            )
        return self
