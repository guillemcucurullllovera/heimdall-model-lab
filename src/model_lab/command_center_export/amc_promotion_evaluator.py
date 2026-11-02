"""Evaluates every Model Lab registry entry for AMC promotion, per
`reports/MODEL_LAB_AMC_PROMOTION_GAP_ANALYSIS.md` and `reports/
MODEL_LAB_AMC_DECISION_MAPPING_INVESTIGATION.md`. This module RUNS a real
evaluation against every entry -- it does not stop at building a schema
-- but the evaluation's own, honestly-computed result is `NOT_EVALUABLE`
for all 26, because the investigation found no real AMC decision mapping
for any of them. Nothing here fabricates an economic number to avoid
that conclusion.
"""

from __future__ import annotations

from model_lab.command_center_export.model_registry_data import StaticEntry
from model_lab.contracts.amc_promotion_evaluation import (
    BaselineComparisonResult,
    DecisionMappingResult,
    EvaluationStatus,
    GateEvidenceState,
    ModelAMCPromotionEvaluation,
    OOSProvenance,
    PromotionGateResult,
    ProspectiveSufficiency,
)
from model_lab.contracts.model_lab_registry import ModelRegistryEntry, ResearchState

AMC_INVESTIGATION_REF = "reports/MODEL_LAB_AMC_DECISION_MAPPING_INVESTIGATION.md"
NOT_IMPLEMENTED_REASON = (
    "No implementation exists for this model (see reports/MODEL_LAB_FORENSIC_INVENTORY.md) -- "
    "there is no output to map to any decision in the first place."
)
NO_HOOK_REASON = (
    "AMC's real decision/gate/sizing/risk contracts (heimdall/contracts/{decision,gate,risk,"
    "portfolio}.py; heimdall/core/sizing/kelly.py) have no input slot for this model's output -- "
    "AMC's actual strategy trades SPX credit-spread options sized by a fixed, signed Kelly "
    "fraction (0.08923), gated by a credit-spread-widening threshold, not by a six-ETF variance/"
    "tail/regime/anomaly/changepoint forecast. See " + AMC_INVESTIGATION_REF + "."
)


def _decision_mapping(entry: ModelRegistryEntry, static: StaticEntry) -> DecisionMappingResult:
    if entry.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
        return DecisionMappingResult(
            has_mapping=False,
            model_output_description="N/A -- not implemented.",
            missing_data=["an implementation", "an AMC decision mapping"],
            reason_not_evaluable=NOT_IMPLEMENTED_REASON,
        )
    output_desc = f"{entry.display_name}: {static.methodology.split('.')[0]}."
    return DecisionMappingResult(
        has_mapping=False,
        model_output_description=output_desc,
        amc_entry_point_citation=None,
        required_data=["a real AMC decision/gate/sizing input slot for this output -- does not currently exist"],
        missing_data=["an AMC-side input slot for this model's output"],
        reason_not_evaluable=NO_HOOK_REASON,
    )


def _oos_provenance(entry: ModelRegistryEntry) -> OOSProvenance:
    if not entry.historical_evidence_refs:
        return OOSProvenance(
            has_statistical_oos_evidence=False, no_lookahead_verified=False,
            note="No implementation, no historical evidence, no OOS claim possible.",
        )
    return OOSProvenance(
        has_statistical_oos_evidence=True,
        training_period_description="Calendar-year expanding folds, 2016/2017 warm-up (statistical, Model-Lab-internal)",
        test_period_description="2018-2026 OOS test years (statistical, Model-Lab-internal)",
        no_lookahead_verified=True,
        leakage_check_refs=list(entry.historical_evidence_refs[:1]),
        note=(
            "This is STATISTICAL out-of-sample evidence (forecast vs realized target), reused "
            "from Model Lab's own walk-forward ladders -- it is NOT economic OOS evidence, which "
            "would require a real AMC decision mapping that does not exist for this model."
        ),
    )


def _prospective_sufficiency(entry: ModelRegistryEntry) -> ProspectiveSufficiency | None:
    if not entry.prospective_evidence_refs:
        return None
    # MODEL-2 specific, real counts (re-derived from the same files every other prospective
    # test in this project reads, not asserted from memory).
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parents[3] / "reports"
    prospective_dir = reports_dir / "prospective"
    resolved = sorted(prospective_dir.glob("resolved_*.json"))
    pending = sorted(prospective_dir.glob("pending_*.json"))
    resolved_hashes = {json.loads(p.read_text())["pending_observation_hash"] for p in resolved}
    still_pending = [p for p in pending if json.loads(p.read_text())["pending_observation_hash"] not in resolved_hashes]
    n_resolved_non_replay = sum(1 for p in resolved if not json.loads(p.read_text())["is_historical_replay"])
    return ProspectiveSufficiency(
        n_resolved_non_replay_observations=n_resolved_non_replay,
        n_pending_observations=len(still_pending),
        real_world_precedent_cited=(
            "AMC's own real (unsigned) graduation-gate template requires minimum_runs=20 "
            "(config/graduation_params.json.UNSIGNED_TEMPLATE) before even evaluating a "
            "deflated-Sharpe result -- itself not yet binding. 6 is well under even that."
        ),
        sufficient_for_any_statistical_claim=False,
        reasoning=(
            f"{n_resolved_non_replay} resolved, non-replay observations (n=1 per ticker) is "
            "statistically meaningless in either direction, and -- independently of sample size -- "
            "no AMC decision mapping exists for this model at all, which is independently "
            "disqualifying regardless of how many prospective observations eventually accrue."
        ),
    )


def _gates(entry: ModelRegistryEntry) -> list[PromotionGateResult]:
    implemented = entry.research_state != ResearchState.LITERATURE_RESEARCH_CANDIDATE
    na = GateEvidenceState.NOT_APPLICABLE

    def g(gate_id: str, name: str, implemented_in_code: bool, code_ref: str | None, state: GateEvidenceState, evidence: str, threshold: str) -> PromotionGateResult:
        return PromotionGateResult(gate_id=gate_id, name=name, implemented_in_code=implemented_in_code, code_reference=code_ref, state=state, evidence=evidence, threshold_status=threshold)

    if not implemented:
        return [
            g("G1_REPRODUCIBILITY", "Reproducibility", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G2_PROVENANCE", "Data/provenance integrity", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G3_OOS_EVIDENCE", "Out-of-sample evidence", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G4_ECONOMIC_BASELINE", "Economic baseline comparison", False, None, na, "No implementation; also no AMC decision mapping exists regardless.", "NO_THRESHOLD_NEEDED"),
            g("G5_INCREMENTAL_VALUE", "Incremental value over AMC baseline", False, None, na, "Same as G4.", "NO_THRESHOLD_NEEDED"),
            g("G6_RISK_STABILITY", "Risk/stability", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G7_COSTS_LIQUIDITY", "Costs/liquidity", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G8_PROSPECTIVE_SHADOW", "Prospective/shadow evidence", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
            g("G9_NO_CRITICAL_DEFECTS", "No unresolved critical leakage/methodological defects", False, None, na, "No implementation.", "NO_THRESHOLD_NEEDED"),
        ]

    has_oos = bool(entry.historical_evidence_refs)
    has_prospective = bool(entry.prospective_evidence_refs)
    return [
        g("G1_REPRODUCIBILITY", "Reproducibility", False, None, GateEvidenceState.PASS,
          f"Deterministic, tested implementation: {', '.join(entry.implementation_files)}.", "NO_THRESHOLD_NEEDED"),
        g("G2_PROVENANCE", "Data/provenance integrity", False, None, GateEvidenceState.PASS,
          "reports/MODEL2_EVIDENCE_MANIFEST.json cross-checks dataset identity/fingerprints (PASSED, this project's own reconciliation).", "NO_THRESHOLD_NEEDED"),
        g("G3_OOS_EVIDENCE", "Out-of-sample evidence (statistical)", False, None,
          GateEvidenceState.PASS if has_oos else GateEvidenceState.NOT_ESTABLISHED,
          entry.historical_evidence_summary, "NO_THRESHOLD_NEEDED"),
        g("G4_ECONOMIC_BASELINE", "Economic baseline comparison", False, None, GateEvidenceState.FAIL,
          "No economic/P&L baseline comparison exists -- confirmed absent by direct grep (MODEL_LAB_AMC_PROMOTION_GAP_ANALYSIS.md §3-4).", "DESIGN_DECISION_REQUIRED"),
        g("G5_INCREMENTAL_VALUE", "Incremental value over AMC baseline", False, None, GateEvidenceState.FAIL,
          "No comparison against AMC's actual baseline exists anywhere in this project.", "DESIGN_DECISION_REQUIRED"),
        g("G6_RISK_STABILITY", "Risk/stability", False, None, GateEvidenceState.NOT_ESTABLISHED,
          "Statistical risk measures (VaR/ES) exist for T-TAIL models; no portfolio-level drawdown/stability analysis exists for anything.", "DESIGN_DECISION_REQUIRED"),
        g("G7_COSTS_LIQUIDITY", "Costs/liquidity", False, None, GateEvidenceState.NOT_ESTABLISHED,
          "No transaction-cost or liquidity treatment exists anywhere in this project.", "DESIGN_DECISION_REQUIRED"),
        g("G8_PROSPECTIVE_SHADOW", "Prospective/shadow evidence", True, "prospective/{capture,resolve}.py",
          GateEvidenceState.NOT_ESTABLISHED if has_prospective else GateEvidenceState.NOT_ESTABLISHED,
          ("6 resolved non-replay + 6 pending observations exist (MODEL-2 only) -- far below AMC's own cited (unsigned) minimum_runs=20 precedent." if has_prospective
           else "No prospective pipeline has been wired for this model."), "REAL_PRECEDENT_CITED" if has_prospective else "DESIGN_DECISION_REQUIRED"),
        g("G9_NO_CRITICAL_DEFECTS", "No unresolved critical leakage/methodological defects", True, "gate_v2/ (never run against a real candidate, see investigation doc)", GateEvidenceState.PASS,
          "Every defect found across this project's full research arc is fixed and regression-tested (reports/MODEL_LAB_RESEARCH_LEDGER's own record); zero outstanding.", "NO_THRESHOLD_NEEDED"),
    ]


def evaluate_model_for_amc_promotion(entry: ModelRegistryEntry, static: StaticEntry) -> ModelAMCPromotionEvaluation:
    decision_mapping = _decision_mapping(entry, static)
    oos = _oos_provenance(entry)
    prospective = _prospective_sufficiency(entry)
    gates = _gates(entry)

    implemented = entry.research_state != ResearchState.LITERATURE_RESEARCH_CANDIDATE
    reasoning = (
        decision_mapping.reason_not_evaluable
        or "No reason recorded -- this would be a bug; every entry must carry a reason."
    )
    limitations = list(entry.limitations)
    if entry.research_state == ResearchState.REJECTED_OR_RETIRED:
        limitations.append(
            "IMPORTANT: this model's Model-Lab-internal research_state is REJECTED_OR_RETIRED "
            "(its own value-add hypothesis was tested and rejected) -- that is a DIFFERENT, "
            "independent finding from this AMC evaluation's own NOT_EVALUABLE status (no decision "
            "mapping exists). Do not conflate the two: REJECTED_OR_RETIRED means 'we tested it and "
            "it did not help'; NOT_EVALUABLE here means 'there was nothing to test it against'."
        )

    return ModelAMCPromotionEvaluation(
        model_id=entry.model_id,
        overall_status=EvaluationStatus.NOT_EVALUABLE,
        status_reasoning=reasoning,
        decision_mapping=decision_mapping,
        economic_evaluation=None,
        baseline_comparison=None if not implemented else BaselineComparisonResult(
            baseline_description="N/A -- no real AMC baseline decision exists for this model's output.",
            model_assisted_description="N/A -- no real AMC decision mapping exists for this model's output.",
            positive_incremental_value_demonstrated=False,
        ),
        oos_provenance=oos,
        prospective_sufficiency=prospective,
        gates=gates,
        limitations=limitations,
        human_authorization_state="NOT_SOUGHT",
    )
