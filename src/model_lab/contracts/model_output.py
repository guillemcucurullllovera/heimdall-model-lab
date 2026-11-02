"""ModelOutput/1 — MODEL1_foundation_contracts.md §6 (M1-C row): "schema
only; the `core` consumer is AMC-side and out of scope." MODEL-1 commits the
schema so a future AMC-side integration (out of scope here, needs separate
authorisation) has a stable contract to target; nothing in this repository
consumes ModelOutput records.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex


class QuantilePoint(ModelLabBaseModel):
    quantile: float = Field(ge=0.0, le=1.0)
    value: str


class ModelOutput(ModelLabBaseModel):
    schema_name: str = "ModelOutput"
    schema_version: int = 1

    model_version_hash: Sha256Hex
    task_id: str
    as_of_utc: datetime
    produced_at_utc: datetime

    target_variable: str
    horizon_days: int = Field(ge=0)

    point_estimate: str | None = None
    quantiles: list[QuantilePoint] = Field(default_factory=list)

    input_hashes: list[Sha256Hex] = Field(default_factory=list)
    decision_relevant: bool = Field(
        default=False, description="True only if this output is bound to a signed prereg/gate ceiling"
    )

    @model_validator(mode="after")
    def _has_some_prediction(self) -> ModelOutput:
        if self.point_estimate is None and not self.quantiles:
            raise ValueError("a ModelOutput must carry at least a point_estimate or quantiles")
        return self
