"""Model Lab Empirical Evaluation / Research Validation Layer contract.

Reuses `ModelLabBaseModel`/`Sha256Hex` directly; does not touch
`amc_promotion_evaluation.py` (which "must remain intact" per instruction)
-- this is a companion contract for the evidence THAT framework's
`EconomicEvaluationResult.None` everywhere was honestly reporting as
absent. The two are cross-referenced, never merged: `amc_promotion_
evaluation.py` answers "does this map to a real AMC decision" (no, for
everything); this contract answers "how would this model's value be
tested empirically on its own terms, independent of AMC" (genuinely, for
every implemented model).

`EvidenceDiscipline` is 6 INDEPENDENT booleans, not one enum -- the
explicit instruction this checkpoint exists to satisfy ("a model can be
implemented without being economically useful... do not collapse these
states").
"""

from __future__ import annotations

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.chart_spec import ChartSpec


class EvidenceDisciplineFlag(ModelLabBaseModel):
    value: bool
    evidence: str


class EvidenceDiscipline(ModelLabBaseModel):
    """6 independent flags. None implies any other -- a model can be
    IMPLEMENTED without being ECONOMICALLY_USEFUL; can be OOS_VALIDATED
    without AMC_ELIGIBLE; can have a backtest (ECONOMICALLY_USEFUL or
    not) without SHADOW_PROSPECTIVE evidence. AMC_ELIGIBLE's `value` MUST
    always be False -- enforced by a test, not merely a convention."""

    implemented: EvidenceDisciplineFlag
    empirically_evaluated: EvidenceDisciplineFlag
    oos_validated: EvidenceDisciplineFlag
    economically_useful: EvidenceDisciplineFlag
    shadow_prospective: EvidenceDisciplineFlag
    amc_eligible: EvidenceDisciplineFlag


class VolTargetingAssumptions(ModelLabBaseModel):
    """Every field here is a DISCLOSED, pre-registered, round-number
    assumption -- never tuned after seeing results, never a measured
    AMC-specific cost."""

    target_annualized_volatility: str
    leverage_cap: str
    cost_bps_per_unit_turnover: str = Field(description="one-way, in basis points, applied to |weight change|")
    risk_free_rate_assumed: str = Field(default="0", description="disclosed simplification -- no risk-free series wired in")
    financing_cost_on_leverage: str = Field(default="not modeled", description="disclosed limitation")
    methodology_reference: str = Field(
        default="Moreira & Muir (2017), 'Volatility-Managed Portfolios', Journal of Finance 72(4):1611-1644"
    )


class VolTargetingResult(ModelLabBaseModel):
    model_id: str
    ticker: str
    assumptions: VolTargetingAssumptions

    n_observations: int
    gross_annualized_return: str
    net_annualized_return: str
    realized_annualized_volatility: str
    sharpe_gross: str
    sharpe_net: str
    max_drawdown: str
    turnover_mean_daily: str
    exposure_mean: str
    exposure_std: str
    worst_day_return: str
    cvar_5pct_daily: str

    benchmark_buy_and_hold: dict = Field(description="same metrics computed for a fixed weight=1 buy-and-hold")
    benchmark_naive_targeted: dict | None = Field(
        default=None, description="same overlay using naive_trailing_rv as the sizing input -- the trivial/classical benchmark"
    )
    incremental_vs_buy_and_hold: dict
    incremental_vs_naive_targeted: dict | None = None

    walk_forward_note: str
    limitations: list[str] = Field(default_factory=list)


class TailCapitalEfficiencyResult(ModelLabBaseModel):
    ticker: str
    level: str
    unconditional_mean_var: str
    conditional_mean_var: str
    combo_mean_var: str | None = None
    pct_change_conditional_vs_unconditional: str
    coverage_note: str


class RiskReductionUtilityResult(ModelLabBaseModel):
    model_id: str
    mean_forward_return_flagged_days: str
    mean_forward_return_unflagged_days: str
    n_flagged: int
    n_unflagged: int
    hypothetical_avoided_loss_if_flat_on_flagged_days: str
    interpretation: str


class ModelEmpiricalEvaluation(ModelLabBaseModel):
    schema_name: str = "ModelEmpiricalEvaluation"
    schema_version: int = 1

    model_id: str
    what_it_estimates: str
    input_data: str
    target_variable: str | None
    horizon_description: str
    output_description: str

    available_data_range: str
    methodology: str
    benchmark_description: str
    incremental_value_question: str
    appropriate_economic_use: str = Field(description="the real downstream use justified by this model's own purpose, or an explicit 'NONE' + why")
    transaction_cost_assumptions: str | None = None

    vol_targeting_results: list[VolTargetingResult] = Field(default_factory=list)
    tail_capital_efficiency: list[TailCapitalEfficiencyResult] = Field(default_factory=list)
    risk_reduction_utility: RiskReductionUtilityResult | None = None

    evidence_discipline: EvidenceDiscipline
    diagnostic_chart_ids: list[str] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)
    reproducibility_command: str
    source_data_refs: list[str] = Field(default_factory=list)


class EmpiricalEvaluationExport(ModelLabBaseModel):
    schema_name: str = "EmpiricalEvaluationExport"
    schema_version: int = 1

    git_commit: str
    evaluations: dict[str, ModelEmpiricalEvaluation]
    not_evaluable: dict[str, str] = Field(description="model_id -> reason, for models with no implementation to evaluate")
    charts: list[ChartSpec] = Field(default_factory=list, description="model-native diagnostic charts -- predicted-vs-realized, calibration, etc.")
    export_sha256: Sha256Hex
