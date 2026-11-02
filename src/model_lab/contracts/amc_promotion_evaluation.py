"""AMC Promotion Evaluation contract -- the missing economic/model-to-AMC
evaluation layer the AMC Promotion Gap Analysis (`reports/MODEL_LAB_AMC_
PROMOTION_GAP_ANALYSIS.md`) found absent for every Model Lab model.

Built AFTER a real, read-only investigation of the actual AMC producer
repository (`reports/MODEL_LAB_AMC_DECISION_MAPPING_INVESTIGATION.md`) --
not invented. That investigation's finding drives this contract's design:
no Model Lab model output currently corresponds to any input field in
AMC's real decision/gate/sizing/risk contracts, for a structural
(architectural) reason, not a data-availability one. Every field below
that would carry a genuine economic evaluation result is `Optional` and
is `None` for every model this pass -- never populated with an invented
number.

STATUS SEMANTICS (`EvaluationStatus`) exist specifically so
"not authorized" can never again be read as "tested and failed": these
are 7 genuinely distinct states, and `NOT_EVALUABLE` -- the state every
one of the 26 registry entries carries this pass -- means "no economic
decision mapping exists to evaluate against," never "evaluated and
found wanting."
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel


class EvaluationStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"  # not yet looked at
    EVALUATION_PENDING = "EVALUATION_PENDING"  # a real evaluation is queued/running
    NOT_EVALUABLE = "NOT_EVALUABLE"  # looked at; no economic decision mapping exists (or no implementation)
    EVALUATED_FAILED = "EVALUATED_FAILED"  # a real evaluation RAN and failed its criteria
    EVALUATED_INCONCLUSIVE = "EVALUATED_INCONCLUSIVE"  # a real evaluation RAN; result inconclusive
    ELIGIBLE_PENDING_HUMAN_AUTHORIZATION = "ELIGIBLE_PENDING_HUMAN_AUTHORIZATION"
    AUTHORIZED = "AUTHORIZED"


class GateEvidenceState(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"  # the model is not far enough along for this gate to apply yet


class DecisionMappingResult(ModelLabBaseModel):
    """Whether, and how, this model's output could enter a real AMC
    decision. `has_mapping=False` for all 26 registry entries this pass
    -- see `MODEL_LAB_AMC_DECISION_MAPPING_INVESTIGATION.md` for why."""

    has_mapping: bool
    model_output_description: str
    amc_entry_point_citation: str | None = Field(
        default=None, description="real AMC file:field this output would enter through, or None"
    )
    decision_it_would_change: str | None = None
    baseline_decision_description: str | None = None
    model_assisted_decision_description: str | None = None
    applicable_portfolio_risk_constraints: list[str] = Field(default_factory=list)
    required_market_option_data: list[str] = Field(default_factory=list)
    execution_assumptions: str | None = None
    required_data: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    reason_not_evaluable: str | None = None


class EconomicEvaluationResult(ModelLabBaseModel):
    """Populated ONLY when a genuine economic evaluation was actually run
    -- every field is `None`/empty for every model this pass, since
    `DecisionMappingResult.has_mapping` is False everywhere. A future
    model with a real decision mapping would have this filled from real
    computation, never from an assumption."""

    cumulative_pnl: str | None = None
    annualized_return: str | None = None
    volatility: str | None = None
    sharpe_ratio: str | None = None
    max_drawdown: str | None = None
    tail_loss_es: str | None = None
    hit_rate: str | None = None
    turnover: str | None = None
    transaction_costs: str | None = None
    capital_usage: str | None = None
    exposure_concentration: str | None = None
    n_observations: int | None = None
    n_independent_events: float | None = None
    regime_breakdown: dict = Field(default_factory=dict)
    options_payoff_note: str | None = Field(
        default=None, description="for options/structured trades: payoff/risk characteristics, never a forced equity-style metric"
    )


class BaselineComparisonResult(ModelLabBaseModel):
    """BASELINE AMC DECISION vs MODEL-ASSISTED AMC DECISION. For every
    implemented model this pass, this is populated with explicit "no
    mapping exists" descriptions and `positive_incremental_value_
    demonstrated=False` -- never a real number, since there is no
    baseline AMC decision to compare against without a decision mapping
    (`DecisionMappingResult.has_mapping=False`). `None` only for
    unimplemented (literature-candidate) entries, where there is nothing
    to even describe."""

    baseline_description: str
    model_assisted_description: str
    incremental_pnl_payoff: str | None = None
    incremental_drawdown: str | None = None
    incremental_risk: str | None = None
    incremental_costs: str | None = None
    incremental_capital_usage: str | None = None
    incremental_tail_behaviour: str | None = None
    incremental_stability: str | None = None
    positive_incremental_value_demonstrated: bool = Field(
        description="MUST be False unless the comparison above actually demonstrates it -- never True by default"
    )


class OOSProvenance(ModelLabBaseModel):
    """Statistical (Model-Lab-internal) out-of-sample provenance --
    DISTINCT from economic OOS (which would require a real decision
    mapping and therefore does not exist for anything this pass)."""

    has_statistical_oos_evidence: bool
    training_period_description: str | None = None
    test_period_description: str | None = None
    no_lookahead_verified: bool
    leakage_check_refs: list[str] = Field(default_factory=list)
    note: str


class ProspectiveSufficiency(ModelLabBaseModel):
    n_resolved_non_replay_observations: int
    n_pending_observations: int
    real_world_precedent_cited: str = Field(
        description="a REAL, already-existing governance precedent cited for context -- never a Model-Lab-invented threshold"
    )
    sufficient_for_any_statistical_claim: bool
    reasoning: str


class PromotionGateResult(ModelLabBaseModel):
    gate_id: str
    name: str
    implemented_in_code: bool = Field(description="True only if real, executable code exists for this gate (cited)")
    code_reference: str | None = None
    state: GateEvidenceState
    evidence: str
    threshold_status: str = Field(
        description="'NO_THRESHOLD_NEEDED' | 'REAL_PRECEDENT_CITED' | 'DESIGN_DECISION_REQUIRED' -- never 'INVENTED'"
    )


class ModelAMCPromotionEvaluation(ModelLabBaseModel):
    schema_name: str = "ModelAMCPromotionEvaluation"
    schema_version: int = 1

    model_id: str
    overall_status: EvaluationStatus
    status_reasoning: str

    decision_mapping: DecisionMappingResult
    economic_evaluation: EconomicEvaluationResult | None = None
    baseline_comparison: BaselineComparisonResult | None = None
    oos_provenance: OOSProvenance
    prospective_sufficiency: ProspectiveSufficiency | None = None
    gates: list[PromotionGateResult]

    limitations: list[str] = Field(default_factory=list)
    human_authorization_state: str = Field(
        default="NOT_SOUGHT", description="'NOT_SOUGHT' for every model this pass -- never anything implying a human was asked"
    )
