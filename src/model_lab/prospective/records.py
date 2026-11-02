"""Local (Model-2-only) record types for genuine prospective-evidence
collection -- deliberately NOT a modification of MODEL-1's frozen
`ModelOutput`/`ModelVersion` contracts (those remain exactly as MODEL-1
committed them; this package only ever produces real, valid instances of
them, same as `shadow/build.py` already does). `ModelOutput` has no field
for "the outcome, once known" -- appending one after the fact would require
mutating an immutable record, which this project's own contract base
(`ModelLabBaseModel`, `frozen=True`) and ledger philosophy (corrections are
new records, never in-place edits) both forbid. These two records are the
append-only companion: a `PendingObservation` wraps the forecasts (never
mutated again); a `ResolvedObservation` is a SEPARATE, later record that
references the pending one by its own content hash and carries the
realized outcome once it exists.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.model_output import ModelOutput


class ForecastEntry(ModelLabBaseModel):
    """One model's forecast, captured alongside the candidate for
    comparison -- the candidate's own `ModelOutput` is kept separately
    (full contract, with provenance); benchmarks are recorded as plain
    (model_id, value) pairs since they do not need their own ModelVersion
    identity for this purpose (they already have one in `shadow/build.py`'s
    own domain; duplicating it here would not add provenance, only noise)."""

    model_id: str
    point_estimate: str


class PendingObservation(ModelLabBaseModel):
    schema_name: str = "PendingObservation"
    schema_version: int = 1

    ticker: str
    task_id: str
    information_cutoff_date: str  # ISO date: the LAST date actually used to produce every forecast below
    information_cutoff_close: str  # the close price ON information_cutoff_date -- needed to compute the
    # realized log return once the NEXT close is resolved, without resolve.py needing to re-load the
    # frozen series (capture and resolve are independent steps, possibly run in different sessions).
    observation_captured_at_utc: datetime  # wall-clock time this record was written

    candidate_output: ModelOutput  # decision_relevant=False, enforced by ModelOutput's own default+this package's code
    benchmark_forecasts: list[ForecastEntry]  # naive/EWMA/GARCH_t/EGARCH_t at the same origin, for comparison

    data_fingerprint: str = Field(description="the frozen DatasetSnapshot's own data_fingerprint, for provenance")
    status: str = Field(default="PENDING", description="'PENDING' always, at construction -- resolution is a separate record")

    pending_observation_hash: Sha256Hex

    @model_validator(mode="after")
    def _status_is_pending(self) -> PendingObservation:
        if self.status != "PENDING":
            raise ValueError("a PendingObservation must be constructed with status='PENDING' -- resolution is a separate, later record")
        return self


def compute_pending_observation_hash(
    ticker: str,
    task_id: str,
    information_cutoff_date: str,
    information_cutoff_close: str,
    candidate_output: ModelOutput,
    benchmark_forecasts: list[ForecastEntry],
    data_fingerprint: str,
) -> str:
    """Self-referential content hash, same pattern as this project's own
    `spec_sha256_of_this_json_excluding_this_field` convention: hashes
    everything ABOVE except the hash field itself, so the hash can be
    independently re-derived and checked by anyone holding the record.

    Deliberately hashes only `candidate_output`'s DETERMINISTIC fields
    (`model_version_hash`, `point_estimate`) rather than its full dump --
    `produced_at_utc` is wall-clock-dependent by design (it records when
    the record was actually built) and must NOT make two captures of the
    IDENTICAL forecast, taken a second apart, hash differently."""
    import json

    payload = {
        "ticker": ticker,
        "task_id": task_id,
        "information_cutoff_date": information_cutoff_date,
        "information_cutoff_close": information_cutoff_close,
        "candidate_model_version_hash": candidate_output.model_version_hash,
        "candidate_point_estimate": candidate_output.point_estimate,
        "benchmark_forecasts": [f.model_dump(mode="json") for f in benchmark_forecasts],
        "data_fingerprint": data_fingerprint,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


class ResolvedObservation(ModelLabBaseModel):
    schema_name: str = "ResolvedObservation"
    schema_version: int = 1

    pending_observation_hash: Sha256Hex  # FK to the PendingObservation this resolves -- never mutates it
    resolved_at_utc: datetime
    realized_target_date: str  # ISO date: the actual trading day whose outcome was observed
    realized_close_prev: str
    realized_close_target: str
    realized_log_return: str
    realized_variance_proxy: str  # squared log return -- the h=1 analogue of forward_realised_variance

    candidate_forecast_error: str = Field(description="QLIKE loss of the candidate's point_estimate vs the realized proxy")
    benchmark_forecast_errors: dict[str, str] = Field(description="model_id -> QLIKE loss, same formula, for comparison")

    resolution_source: str = Field(description="e.g. 'massive_live_fetch' -- always disclosed, never silent")
    is_historical_replay: bool = Field(
        default=False,
        description="True only if this resolution reused data already present in the frozen "
        "research dataset at capture time (would indicate a bug in this pipeline's own design, "
        "since capture only ever targets dates NOT yet in that dataset -- kept as an explicit, "
        "checkable field rather than an assumption)",
    )
