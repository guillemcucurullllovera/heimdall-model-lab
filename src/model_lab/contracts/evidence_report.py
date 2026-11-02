"""EvidenceReport/1 — bundles a Gate v2 verdict and scoring results for one
(ModelVersion, Task) pair into the record consumed by publish/ (ChartSpec
builders read from this). Named as a ledger `record_type` in
MODEL1_foundation_contracts.md §12 and as one of the pydantic models
committed as JSON Schema in §6 (M1-C row); no field-level schema is spelled
out verbatim in the contracts doc, so this is designed to satisfy every
requirement §14 and §18 place on "every evaluation output": n, n_independent,
MDE, data cutoff beside every performance number, plus the gate verdict and
its full identity fields so a report can never be mistaken for a different
gate version/config/code state.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.chart_spec import Causality


class ScoreResult(ModelLabBaseModel):
    name: str
    value: str = Field(description="stringified; see canon/jcs.py's float policy")
    n: int
    n_independent: float
    mde: str | None = None
    data_cutoff: date | None = None
    causality: Causality = Causality.NOT_APPLICABLE


class GuardResult(ModelLabBaseModel):
    name: str
    passed: bool
    margin_observed: str | None = None
    margin_required: str


class EvidenceReport(ModelLabBaseModel):
    schema_name: str = "EvidenceReport"
    schema_version: int = 1

    report_id: str
    model_version_hash: Sha256Hex
    task_id: str
    prereg_id: str
    run_id: str
    as_of_utc: datetime

    primary_score_result: ScoreResult
    secondary_score_results: list[ScoreResult] = Field(default_factory=list)
    guard_results: list[GuardResult] = Field(default_factory=list)

    gate_result_hash: Sha256Hex | None = Field(
        default=None, description="hash of the GateResult record this report was judged against, if any"
    )
    gate_verdict: str | None = Field(default=None, description="'SUFFICIENT' | 'INSUFFICIENT' | 'FAIL' | None")

    report_sha256: Sha256Hex
