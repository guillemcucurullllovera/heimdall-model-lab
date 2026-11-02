"""Prereg/1 and RunManifest/1 — MODEL1_foundation_contracts.md §10.

Rules: prereg precedes any run; edits create a new prereg that inherits N;
every scored configuration (including failed/pruned) is a `trial_batch`
record; a prereg whose MDE at available n exceeds any plausible effect may
proceed only as `diagnostic_evidence_only`; a prereg against a BLOCKED task
is rejected by the ledger (enforced by the ledger append path, not here —
see ledger/governance.py).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.task import GuardSpec, InfluenceTier, ScoreSpec


class DecisionRelevance(StrEnum):
    CONFIRMATORY = "confirmatory"
    EXPLORATORY_MARGINAL = "exploratory_marginal"
    DIAGNOSTIC_EVIDENCE_ONLY = "diagnostic_evidence_only"
    NOT_DECIDABLE = "not_decidable"


class SearchSpaceSpec(ModelLabBaseModel):
    hash: Sha256Hex
    max_trials: int = Field(ge=1)
    parameters: dict[str, list[float | int | str]] = Field(
        default_factory=dict, description="parameter_name -> range/choice list"
    )
    seeds: list[int] = Field(default_factory=list)


class Prereg(ModelLabBaseModel):
    schema_name: str = "Prereg"
    schema_version: int = 1

    prereg_id: str
    lineage_id: str
    N_inherited: int = Field(ge=0)
    hypothesis: str
    mechanism: str
    task_ids: list[str] = Field(min_length=1)
    model_family: str

    incumbent_ids: list[str] = Field(default_factory=list)
    naive_id: str | None = None
    primary_score: ScoreSpec
    secondary_scores: list[ScoreSpec] = Field(default_factory=list)
    guards: list[GuardSpec] = Field(default_factory=list)

    search_space: SearchSpaceSpec
    splitter_config_id: str
    data_snapshot_ids: list[str]

    mde_at_available_n: str = Field(description="the MDE statement this prereg is judged against")
    decision_relevance: DecisionRelevance
    amc_entry_ceiling: InfluenceTier
    kill_criteria: list[str] = Field(min_length=1, description="pre-declared; must include at least one")
    forbidden_analyses: list[str] = Field(default_factory=list)

    author_id: str
    contributors: list[str] = Field(default_factory=list)
    created_at: datetime
    prereg_sha256: Sha256Hex


class DeterminismTier(StrEnum):
    BITWISE = "BITWISE"
    NUMERIC = "NUMERIC"
    STATISTICAL = "STATISTICAL"


class RunManifest(ModelLabBaseModel):
    schema_name: str = "RunManifest"
    schema_version: int = 1

    run_id: str
    prereg_id: str
    experiment_fingerprint: Sha256Hex
    code_tree_hash: Sha256Hex
    dirty: bool = Field(description="must be False for evidence-grade runs")
    env_lock_hash: Sha256Hex
    blas_threads: int = Field(ge=1)
    cpu_flags: list[str] = Field(default_factory=list)
    seed_tree: dict[str, int] = Field(default_factory=dict)
    dataset_hashes: list[Sha256Hex]
    output_hashes: list[Sha256Hex]
    determinism_tier: DeterminismTier
    wall_seconds: float = Field(ge=0)
    cores: int = Field(ge=1)
    peak_mem_bytes: int = Field(ge=0)
