"""Model Lab multi-model registry/export contract — the Command-Center
productization checkpoint beyond MODEL-2 alone (see `reports/
MODEL_LAB_FORENSIC_INVENTORY.md` for the inventory this contract projects).

Same design discipline as `model2_export.py`: NOT built on `EvidenceReport`/
`PublicationManifest` (Gate-v2/ledger-coupled fields this repository's
actual current state cannot honestly fill for most of these entries),
reuses `ModelLabBaseModel`/`Sha256Hex`/`LifecycleState`/`ChartSpec`
directly wherever no fabrication is required, and uses explicit
`EstablishmentState`/`NOT_ESTABLISHED` sentinels everywhere a real current
value does not exist.

`ResearchState` is the five-state classification this checkpoint's own
instruction required (`IMPLEMENTED_REPRODUCIBLE` through
`REJECTED_OR_RETIRED`) -- every model entry carries exactly one, and
nothing in this contract allows hiding categories 2-5 (there is no
"promoted" bucket that silently excludes a REJECTED_OR_RETIRED entry; it
remains a first-class, fully-populated entry, same as every other state).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from model_lab.contracts.amc_promotion_evaluation import ModelAMCPromotionEvaluation
from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.chart_spec import ChartSpec
from model_lab.contracts.model2_export import EstablishmentState


class ResearchState(StrEnum):
    IMPLEMENTED_REPRODUCIBLE = "IMPLEMENTED_REPRODUCIBLE"
    IMPLEMENTED_VALIDATION_PENDING = "IMPLEMENTED_VALIDATION_PENDING"
    SHADOW_RESEARCH = "SHADOW_RESEARCH"
    LITERATURE_RESEARCH_CANDIDATE = "LITERATURE_RESEARCH_CANDIDATE"
    REJECTED_OR_RETIRED = "REJECTED_OR_RETIRED"


class RegistryBucket(StrEnum):
    """UI-grouping hint, cross-cutting the task hierarchy. A model can be
    both grouped under its task (via `task_id`) AND listed in one of these
    cross-cutting buckets -- the two views are complementary, not
    exclusive (e.g. a `REJECTED_RETIRED` regime model still belongs under
    T-STATE in the task tree)."""

    TASK_MODEL = "TASK_MODEL"
    COMPOSITE_CHALLENGER = "COMPOSITE_CHALLENGER"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    SHADOW = "SHADOW"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    REJECTED_RETIRED = "REJECTED_RETIRED"


class ModelRegistryEntry(ModelLabBaseModel):
    model_id: str
    display_name: str
    family: str
    research_state: ResearchState
    bucket: RegistryBucket

    task_id: str | None = Field(description="the frozen Task this model belongs to, if any")
    constituent_of: str | None = Field(default=None, description="model_id of a composite this is a constituent of")
    constituents: list[str] = Field(default_factory=list, description="model_ids of this composite's own constituents")

    target_variable: str | None = None
    universe: list[str] = Field(default_factory=list)
    horizons_days: list[int] = Field(default_factory=list)
    methodology: str
    inputs_data: str

    implementation_files: list[str] = Field(default_factory=list)
    code_fingerprint: Sha256Hex | None = None
    source_references: list[str] = Field(default_factory=list, description="real paper/corpus filenames or citations")

    lifecycle_state: str | None = Field(default=None, description="a real LifecycleState value, if any ModelVersion exists")
    independent_review_state: EstablishmentState
    amc_promotion_authorized: bool = Field(description="MUST always be False; enforced by a test")

    historical_evidence_summary: str
    historical_evidence_refs: list[str] = Field(default_factory=list)
    oos_evidence_summary: str | None = None
    prospective_evidence_summary: str | None = None
    prospective_evidence_refs: list[str] = Field(default_factory=list)

    actual_outputs_sample: dict = Field(
        default_factory=dict, description="a small, REAL sample of what this model actually produces -- never a description-only placeholder"
    )
    benchmark_comparison_summary: str | None = None
    incremental_value_summary: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    governance_provenance: list[str] = Field(default_factory=list)

    chart_ids: list[str] = Field(default_factory=list)
    result_artifact_references: list[str] = Field(default_factory=list)


class ResearchSummaryCounts(ModelLabBaseModel):
    n_implemented_reproducible: int
    n_implemented_validation_pending: int
    n_shadow_research: int
    n_literature_research_candidates: int
    n_rejected_or_retired: int
    n_task_families_represented: int
    n_models_with_historical_evidence: int
    n_models_with_prospective_evidence: int
    n_models_with_charts: int
    n_models_amc_eligible: int = Field(description="MUST always be 0")
    n_models_amc_not_evaluable: int = Field(description="models with no real AMC decision mapping (or no implementation)")
    n_models_amc_evaluated_failed: int = Field(description="MUST always be 0 this pass -- no real evaluation has ever run")


class ModelLabRegistryExport(ModelLabBaseModel):
    schema_name: str = "ModelLabRegistryExport"
    schema_version: int = 1

    generated_at_utc: datetime
    git_commit: str
    git_branch: str

    entries: list[ModelRegistryEntry]
    task_hierarchy: dict[str, list[str]] = Field(description="task_id -> ordered list of model_ids under it")
    status_buckets: dict[str, list[str]] = Field(description="RegistryBucket value -> list of model_ids")

    amc_promotion_evaluations: dict[str, ModelAMCPromotionEvaluation] = Field(
        description="model_id -> its AMC promotion evaluation (see contracts/amc_promotion_evaluation.py)"
    )

    charts: list[ChartSpec] = Field(default_factory=list)
    research_summary: ResearchSummaryCounts

    export_sha256: Sha256Hex
