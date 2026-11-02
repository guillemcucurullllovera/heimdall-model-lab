"""OptionSnapshotRecord/1 — MODEL1_foundation_contracts.md §18a (verbatim schema).

Append-only; `prev_hash` chained; one record per contract observation.
Rules (verbatim numbering, §18a):
1. append-only with prev_hash chaining and periodic operator-signed checkpoints.
2. raw responses stored content-addressed and unmodified.
3. `quote_time_utc` stored only if the vendor supplies it; a record without one
   is `replay_grade = FETCH_TIME_ONLY` (forward diagnostics only, never
   no-hindsight replay).
4. IV/greeks are vendor-authoritative only when the vendor documents the
   model; otherwise Model Lab's own solver, with the model version recorded.
5. rate/dividend assumptions are explicit fields with their sources.
6. a source with licence.status != CLEAR may be archived only if the AMC
   already ingests it and the operator accepted that internal-archive risk;
   any dataset built from it is research_only_unlicensed.
7. no source is added by MODEL-1 (observer-side ingestion of what the AMC
   already persists, only).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.licence import LicenceStatus


class ObservationKind(StrEnum):
    AMC_DISCOVERY = "AMC_DISCOVERY"
    AMC_VALIDATION = "AMC_VALIDATION"
    OTHER = "OTHER"


class AcquisitionStatus(StrEnum):
    OK = "OK"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    STALE = "STALE"
    ENTITLEMENT_DENIED = "ENTITLEMENT_DENIED"


class ReplayGrade(StrEnum):
    """Whether a vendor-supplied quote_time_utc is present. NOTE: this is a
    narrow, mechanical signal (rule 3 of §18a) — it says a genuine
    no-hindsight replay is *technically possible* for this record because we
    know when the quote was actually taken, not that the archive as a whole
    is sufficient for a no-hindsight replay *study* (other limitations —
    single-date coverage, no later quotes to compare against, indicative
    vs NBBO — are tracked separately via data_quality_flags and
    licence.permitted_use_note)."""

    VENDOR_QUOTE_TIMESTAMPED = "VENDOR_QUOTE_TIMESTAMPED"
    FETCH_TIME_ONLY = "FETCH_TIME_ONLY"


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    AMERICAN = "american"
    EUROPEAN = "european"


class OptionLicenceInfo(ModelLabBaseModel):
    status: LicenceStatus
    basis_url: str | None = None
    checked_utc: datetime
    permitted_use_note: str


class UnderlyingObservation(ModelLabBaseModel):
    symbol: str
    price: str
    price_source: str
    price_time_utc: datetime
    price_age_s: int | None = Field(
        default=None, description="whole seconds; sub-second precision is not needed and floats can't be JCS-hashed (see canon/jcs.py)"
    )


class OptionContractSpec(ModelLabBaseModel):
    occ_symbol: str
    expiry: date
    strike: str
    right: OptionRight
    style: ExerciseStyle
    multiplier: int = 100
    deliverable_notes: str | None = None


class OptionQuote(ModelLabBaseModel):
    bid: str | None = None
    ask: str | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last: str | None = None
    last_time_utc: datetime | None = None
    quote_time_utc: datetime | None = Field(
        default=None, description="only set if the vendor supplies it (rule 3); never inferred"
    )
    quote_age_s: int | None = Field(
        default=None, description="whole seconds; see UnderlyingObservation.price_age_s for why not float"
    )
    exchange_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _quote_time_and_age_together(self) -> OptionQuote:
        if self.quote_time_utc is None and self.quote_age_s is not None:
            raise ValueError("quote_age_s must be null whenever quote_time_utc is null (rule 3)")
        return self


class MidDerivation(ModelLabBaseModel):
    mid: str | None = None
    derived: bool = Field(description="True if computed as (bid+ask)/2; False if unavailable")
    crossed: bool = False


class MarketState(ModelLabBaseModel):
    volume: int | None = None
    open_interest: int | None = None
    oi_as_of_utc: datetime | None = None


class VendorGreeks(ModelLabBaseModel):
    iv: str | None = None
    delta: str | None = None
    gamma: str | None = None
    theta: str | None = None
    vega: str | None = None
    rho: str | None = None
    model: str | None = Field(default=None, description="vendor-documented pricing model, if named")
    as_of: datetime | None = None


class DerivedAnalytics(ModelLabBaseModel):
    iv_mid_bs: str | None = None
    iv_solver_version: str | None = None
    forward: str | None = None
    rate_source: str | None = None
    rate: str | None = None
    dividend_yield_or_schedule: str | None = None
    dividend_source: str | None = None
    greeks_own: dict[str, str] | None = None
    american_adjustment_method: str | None = None


class DataQualityFlag(StrEnum):
    CROSSED = "crossed"
    LOCKED = "locked"
    ZERO_BID = "zero_bid"
    WIDE_SPREAD = "wide_spread"
    STALE_QUOTE = "stale_quote"
    MISSING_IV = "missing_iv"
    EXTREME_IV = "extreme_iv"
    OFF_MARKET_UNDERLYING = "off_market_underlying"


class RawRef(ModelLabBaseModel):
    content_addressed_path: str
    byte_length: int = Field(ge=0)


class OptionSnapshotRecord(ModelLabBaseModel):
    schema_name: str = "OptionSnapshotRecord"
    schema_version: int = 1

    archive_seq: int = Field(ge=0)
    prev_hash: Sha256Hex | None = None
    record_hash: Sha256Hex

    observation_utc: datetime
    observation_kind: ObservationKind
    amc_experiment_id: str | None = None
    amc_cycle_id: str | None = None

    provider_id: str
    provider_feed: str
    request_fingerprint: str | None = None
    response_fingerprint: Sha256Hex = Field(description="sha256 of raw bytes")
    normalized_rows_sha256: Sha256Hex

    acquisition_status: AcquisitionStatus
    licence: OptionLicenceInfo

    underlying: UnderlyingObservation
    contract: OptionContractSpec
    quote: OptionQuote
    mid: MidDerivation
    market_state: MarketState
    vendor_greeks: VendorGreeks | None = None
    derived: DerivedAnalytics | None = None
    data_quality_flags: list[DataQualityFlag] = Field(default_factory=list)
    raw_ref: RawRef

    replay_grade: ReplayGrade = Field(
        description="FETCH_TIME_ONLY unless quote.quote_time_utc is vendor-supplied (rule 3)"
    )

    @model_validator(mode="after")
    def _replay_grade_matches_quote_time(self) -> OptionSnapshotRecord:
        expected = (
            ReplayGrade.VENDOR_QUOTE_TIMESTAMPED
            if self.quote.quote_time_utc is not None
            else ReplayGrade.FETCH_TIME_ONLY
        )
        if self.replay_grade != expected:
            raise ValueError(
                f"replay_grade {self.replay_grade!r} inconsistent with quote.quote_time_utc "
                f"(expected {expected!r})"
            )
        return self


class OptionArchiveChainManifest(ModelLabBaseModel):
    """One per ingested snapshot batch."""

    schema_name: str = "OptionArchiveChainManifest"
    schema_version: int = 1

    manifest_id: str
    n_contracts: int = Field(ge=0)
    coverage_vs_expected_chain: str = Field(description="fraction or 'unknown' if partial")
    calendar_holiday_flags: list[str] = Field(default_factory=list)
    ingest_code_hash: Sha256Hex
    source_licence_snapshot_hash: Sha256Hex
    archive_head_hash: Sha256Hex
