"""MODEL-2 Command-Center export/read-model contract.

Deliberately NOT built on `EvidenceReport`/`PublicationManifest`
(`chart_spec.py`, `evidence_report.py`): those contracts are Gate-v2-
coupled (`EvidenceReport.prereg_id` is a required, non-optional field
identifying a real Gate v2 pre-registration; `PublicationManifest.
ledger_head_hash` is a required field identifying a real position in the
hash-chained evidence ledger). The MODEL-2 candidate has gone through
NEITHER — it has never been pre-registered against Gate v2, and no shadow
or prospective record this project has produced has ever been written to
the real ledger (`shadow/__init__.py`'s own docstring: "no write to the
real hash-chained evidence ledger"). Using those contracts here would
require fabricating a `prereg_id` or `ledger_head_hash` that does not
genuinely exist — exactly what this export must never do.

This module instead defines a narrower, honestly-scoped contract for
exactly what DOES genuinely exist: the candidate's locked specification,
its real historical/prospective/downstream-value/confidence evidence, its
research ledger, and its promotion-gate status, each sourced from already-
committed evidence rather than re-derived prose. `ChartSpec` IS reused
(§ below) for the one place it fits without any fabrication — plotting
real QLIKE comparisons, which carries no Gate-v2/ledger coupling at all.

Every field that has no genuine value yet uses `NOT_ESTABLISHED` or
`PENDING` (see `EstablishmentState`) rather than a placeholder that could
be mistaken for a real value -- the Command Center consumer must be able
to render "not established" distinctly from "zero" or "passed", mirroring
`ChartSpec`'s own `unavailable`-never-renders-as-zero discipline.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.chart_spec import ChartSpec
from model_lab.contracts.model_version import LifecycleState


class EstablishmentState(StrEnum):
    """Used wherever a value may genuinely not exist yet. Never silently
    omitted -- a field using this enum is always PRESENT in the export,
    just carrying this instead of a fabricated real value."""

    ESTABLISHED = "ESTABLISHED"
    PENDING = "PENDING"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class GateState(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class DecisionLabel(StrEnum):
    RETAIN = "RETAIN"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    MIXED = "MIXED"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    DEFINITIONAL = "DEFINITIONAL"
    CAPABILITY_BUILT = "CAPABILITY_BUILT"
    DEFECT_FOUND_AND_FIXED = "DEFECT_FOUND_AND_FIXED"
    VERIFIED_CLEAN = "VERIFIED_CLEAN"


# ---- 1. Model identity ----------------------------------------------------


class ModelIdentitySection(ModelLabBaseModel):
    model_id: str
    family: str
    lifecycle_state: LifecycleState
    git_commit: str
    git_branch: str
    working_tree_dirty: bool
    candidate_code_fingerprint: Sha256Hex = Field(
        description="sha256 of the concatenated, sorted source files that DEFINE the candidate "
        "(vol/combination.py, vol/candidate_spec.py, vol/garch_models.py, vol/baselines.py) -- "
        "identifies THIS version of the candidate independent of which ticker/day it is run "
        "against, unlike a per-run ModelVersion.model_version_hash (which depends on that run's "
        "own fold parameters)."
    )
    per_ticker_shadow_model_version_hash: dict[str, Sha256Hex] = Field(
        description="model_id (vol/garch_models.py-style, PER TICKER) -> the real "
        "ModelVersion.model_version_hash from the most recent shadow snapshot run "
        "(shadow/build_combo.py); empty if no snapshot has been taken for that ticker yet."
    )
    generated_at_utc: datetime


# ---- 2. Model purpose/family ----------------------------------------------


class ModelPurposeSection(ModelLabBaseModel):
    task_id: str
    task_status: str = Field(description="the real, frozen Task.status value (tasks/registry.py)")
    target_variable: str
    horizons_days: list[int]
    universe: list[str]
    primary_score: str
    naive_benchmark: str | None
    frozen_incumbents: list[str] = Field(description="T-VOL's own frozen incumbents list (tasks/registry.py)")
    candidate_constituents: tuple[str, ...]
    candidate_combination_rule: str
    candidate_strongest_benchmark: str
    challenger_relationship: str = Field(
        description="factual, non-promotional description of what kind of challenger this is"
    )


# ---- 3. Candidate definition ----------------------------------------------


class CandidateCellRef(ModelLabBaseModel):
    ticker: str
    horizon: int


class CandidateDefinitionSection(ModelLabBaseModel):
    eligible_cells: list[CandidateCellRef]
    rejected_cells: list[CandidateCellRef]
    information_set: str
    training_procedure: str
    output_description: str
    spec_document: str = Field(default="reports/MODEL2_CANDIDATE_SPEC.md")


# ---- 4. Historical / walk-forward evidence --------------------------------


class HistoricalCellResult(ModelLabBaseModel):
    ticker: str
    horizon: int
    candidate_qlike: str
    benchmark_qlike: str
    benchmark_model_id: str
    dm_p_value: str
    n_oos: int
    is_eligible: bool


class HistoricalEvidenceSection(ModelLabBaseModel):
    cells: list[HistoricalCellResult]
    n_cells_significant_better: int
    n_cells_significant_worse: int
    n_cells_no_significant_difference: int
    negative_findings: list[str]
    regime_limitations: list[str]
    source_files: list[str]


# ---- 5. Prospective evidence -----------------------------------------------


class ProspectiveObservationRef(ModelLabBaseModel):
    ticker: str
    information_cutoff_date: str
    status: str = Field(description="'PENDING' or 'RESOLVED'")
    resolved_target_date: str | None = None
    candidate_forecast_error: str | None = None
    is_historical_replay: bool | None = None
    pending_observation_hash: Sha256Hex


class ProspectiveEvidenceSection(ModelLabBaseModel):
    n_resolved_non_replay_observations: int
    n_pending_observations: int
    observations: list[ProspectiveObservationRef]
    sufficiency_state: EstablishmentState = Field(
        description="whether enough prospective evidence exists for a promotion-relevant "
        "statistical claim -- see promotion_gates section; NOT_ESTABLISHED at small N by design"
    )
    historical_vs_prospective_note: str


# ---- 6. Downstream value ---------------------------------------------------


class DownstreamValueCell(ModelLabBaseModel):
    ticker: str
    tail_level: str
    incumbent_christoffersen_p: str
    candidate_christoffersen_p: str
    candidate_improved: bool
    is_eligible_ticker: bool


class DownstreamValueSection(ModelLabBaseModel):
    cells: list[DownstreamValueCell]
    pre_registered_rule: str
    mechanical_decision: DecisionLabel
    substantive_caveat: str = Field(
        description="the honest, non-mechanical reading -- e.g. most cells remain miscalibrated "
        "regardless of the mechanical RETAIN verdict"
    )
    source_file: str = Field(default="reports/downstream_value_probe/_summary.json")


# ---- 7. Confidence / abstention --------------------------------------------


class ConfidenceAbstentionSection(ModelLabBaseModel):
    per_ticker_top_tertile_loss: dict[str, str]
    per_ticker_bottom_tertile_loss: dict[str, str]
    pooled_dm_stat: str
    pooled_dm_p_value: str
    decision: DecisionLabel
    limitations: list[str]
    source_file: str = Field(default="reports/confidence_abstention_probe/_summary.json")


# ---- 8. Research ledger ----------------------------------------------------


class ResearchLedgerEntry(ModelLabBaseModel):
    index: int
    phase: str
    experiment: str
    decision: DecisionLabel
    reason: str
    evidence: str


class ResearchLedgerSection(ModelLabBaseModel):
    entries: list[ResearchLedgerEntry]
    n_retain: int
    n_reject_or_inconclusive: int
    n_descriptive_or_definitional: int
    source_document: str = Field(default="reports/MODEL2_RESEARCH_LEDGER.md")


# ---- 9. Promotion gates -----------------------------------------------------


class PromotionGateEntry(ModelLabBaseModel):
    gate_id: str
    name: str
    state: GateState
    evidence: str
    maps_to_lifecycle_state: str | None = None


class PromotionGatesSection(ModelLabBaseModel):
    gates: list[PromotionGateEntry]
    amc_promotion_authorized: bool = Field(description="MUST always be False; enforced by a test")
    source_document: str = Field(default="reports/MODEL2_PROMOTION_GATES.md")


# ---- 10. Provenance ---------------------------------------------------------


class ProvenanceSection(ModelLabBaseModel):
    evidence_manifest_file: str = Field(default="reports/MODEL2_EVIDENCE_MANIFEST.json")
    evidence_manifest_sha256: Sha256Hex
    test_suite_command: str
    test_suite_last_known_result: str
    literature_references: list[str]
    reproduction_commands: list[str]


# ---- 11. Health / limitations -----------------------------------------------


class HealthLimitationsSection(ModelLabBaseModel):
    known_defects_outstanding: list[str]
    defects_found_and_fixed_this_arc: list[str]
    data_limitations: list[str]
    prospective_evidence_limitations: list[str]
    model_specific_failure_modes: list[str]
    independent_review_status: EstablishmentState


# ---- Top level ---------------------------------------------------------------


class Model2CommandCenterExport(ModelLabBaseModel):
    schema_name: str = "Model2CommandCenterExport"
    schema_version: int = 1

    identity: ModelIdentitySection
    purpose: ModelPurposeSection
    candidate_definition: CandidateDefinitionSection
    historical_evidence: HistoricalEvidenceSection
    prospective_evidence: ProspectiveEvidenceSection
    downstream_value: DownstreamValueSection
    confidence_abstention: ConfidenceAbstentionSection
    research_ledger: ResearchLedgerSection
    promotion_gates: PromotionGatesSection
    provenance: ProvenanceSection
    health_limitations: HealthLimitationsSection

    charts: list[ChartSpec] = Field(default_factory=list)

    export_sha256: Sha256Hex = Field(
        description="self-referential: sha256 of this document's own canonical JSON, "
        "EXCLUDING this field -- see command_center_export/builder.py for computation"
    )
