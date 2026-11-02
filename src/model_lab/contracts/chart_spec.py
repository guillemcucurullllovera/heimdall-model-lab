"""ChartSpec/1 and the publication directory contract —
MODEL1_foundation_contracts.md §18 (cites reports/command_center_evidence_design.md §2-3).

"immutable `evidence.json`, `charts/*.json`, `manifest.json` (SHA-256 of
every file + ledger head + model version hash), `model_card.json`,
`datasheet.json`; `data_authority = PROVISIONAL` only when hashes recompute;
each chart names its decision question, shows the incumbent, carries bands,
prints `n`/MDE/data cutoff/causality (filtered vs smoothed), and renders
`UNAVAILABLE` instead of a zero."
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex


class Causality(StrEnum):
    FILTERED = "FILTERED"
    SMOOTHED = "SMOOTHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataAuthority(StrEnum):
    PROVISIONAL = "PROVISIONAL"
    UNAVAILABLE = "UNAVAILABLE"


class ChartSeriesPoint(ModelLabBaseModel):
    """A single plotted value. `unavailable=True` renders as UNAVAILABLE in
    the Command Center rather than as a zero — `value` must be None
    whenever `unavailable` is True, and non-None otherwise."""

    x_label: str
    value: str | None = Field(default=None, description="stringified to avoid float-canonicalization issues")
    band_lower: str | None = None
    band_upper: str | None = None
    unavailable: bool = False

    @model_validator(mode="after")
    def _unavailable_consistency(self) -> ChartSeriesPoint:
        if self.unavailable and self.value is not None:
            raise ValueError("unavailable points must not carry a value (never render UNAVAILABLE as 0)")
        if not self.unavailable and self.value is None:
            raise ValueError("a point that is not unavailable must carry a value")
        return self


class ChartSeries(ModelLabBaseModel):
    label: str
    role: str = Field(description="'challenger' | 'incumbent' | 'naive' | 'comparator'")
    points: list[ChartSeriesPoint]


class ChartSpec(ModelLabBaseModel):
    schema_name: str = "ChartSpec"
    schema_version: int = 1

    chart_id: str
    decision_question: str
    incumbent_id: str
    series: list[ChartSeries]

    n: int | None = None
    n_independent: float | None = None
    mde: str | None = None
    data_cutoff: date | None = None
    causality: Causality
    data_authority: DataAuthority

    chart_sha256: Sha256Hex

    @model_validator(mode="after")
    def _incumbent_series_present(self) -> ChartSpec:
        roles = {s.role for s in self.series}
        if "incumbent" not in roles:
            raise ValueError("a ChartSpec must include an 'incumbent' series (§18: 'shows the incumbent')")
        return self


class PublicationManifest(ModelLabBaseModel):
    """manifest.json: SHA-256 of every published file + ledger head + model
    version hash. `data_authority = PROVISIONAL` is asserted only when every
    listed file's hash recomputes at read time (checked by
    publish/verify.py, not here)."""

    schema_name: str = "PublicationManifest"
    schema_version: int = 1

    model_id: str
    model_version_hash: Sha256Hex
    ledger_head_hash: Sha256Hex
    published_at_utc: datetime
    file_hashes: dict[str, Sha256Hex] = Field(description="relative path -> sha256 of file bytes")
    data_authority: DataAuthority
