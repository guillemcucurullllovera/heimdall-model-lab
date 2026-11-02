from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.command_center_export.model_registry_data import STATIC_ENTRIES
from model_lab.command_center_export.registry_builder import build_model_lab_registry
from model_lab.contracts.amc_promotion_evaluation import EvaluationStatus, GateEvidenceState
from model_lab.contracts.model_lab_registry import ResearchState

REPO_ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _registry():
    return build_model_lab_registry(NOW)


def _static_by_id():
    return {s.model_id: s for s in STATIC_ENTRIES}


def test_not_authorized_is_a_distinct_enum_value_from_evaluated_failed() -> None:
    """The whole point of this checkpoint: NOT_EVALUABLE/NOT_AUTHORIZED must
    never collapse onto the same representation as EVALUATED_FAILED."""
    assert EvaluationStatus.NOT_EVALUABLE != EvaluationStatus.EVALUATED_FAILED
    assert EvaluationStatus.NOT_EVALUABLE != EvaluationStatus.EVALUATED_INCONCLUSIVE
    all_values = {s.value for s in EvaluationStatus}
    assert all_values == {
        "NOT_EVALUATED", "EVALUATION_PENDING", "NOT_EVALUABLE", "EVALUATED_FAILED",
        "EVALUATED_INCONCLUSIVE", "ELIGIBLE_PENDING_HUMAN_AUTHORIZATION", "AUTHORIZED",
    }


def test_no_model_is_evaluated_failed_this_pass() -> None:
    """EVALUATED_FAILED means a real evaluation RAN and failed. Since no
    real AMC decision mapping exists for anything, nothing may carry this
    status -- it would misrepresent NOT_EVALUABLE as a completed, failed
    test."""
    reg = _registry()
    for ev in reg.amc_promotion_evaluations.values():
        assert ev.overall_status != EvaluationStatus.EVALUATED_FAILED
        assert ev.overall_status == EvaluationStatus.NOT_EVALUABLE
    assert reg.research_summary.n_models_amc_evaluated_failed == 0


def test_no_promotion_without_economic_evidence() -> None:
    reg = _registry()
    for ev in reg.amc_promotion_evaluations.values():
        assert ev.overall_status not in (EvaluationStatus.ELIGIBLE_PENDING_HUMAN_AUTHORIZATION, EvaluationStatus.AUTHORIZED)
        assert ev.economic_evaluation is None
    assert reg.research_summary.n_models_amc_eligible == 0


def test_no_promotion_from_in_sample_or_statistical_oos_evidence_alone() -> None:
    """Even a model with real statistical OOS evidence (e.g. GARCH_t) must
    not be promoted purely on that basis -- economic evidence is a
    separate, also-required gate."""
    reg = _registry()
    ev = reg.amc_promotion_evaluations["GARCH_t"]
    assert ev.oos_provenance.has_statistical_oos_evidence is True
    assert ev.overall_status == EvaluationStatus.NOT_EVALUABLE
    economic_gate = next(g for g in ev.gates if g.gate_id == "G4_ECONOMIC_BASELINE")
    assert economic_gate.state == GateEvidenceState.FAIL


def test_no_promotion_from_insufficient_prospective_sample() -> None:
    reg = _registry()
    ev = reg.amc_promotion_evaluations["Combo_median_EWMA_GARCH_EGARCH"]
    assert ev.prospective_sufficiency is not None
    assert ev.prospective_sufficiency.n_resolved_non_replay_observations == 6
    assert ev.prospective_sufficiency.sufficient_for_any_statistical_claim is False
    assert ev.overall_status == EvaluationStatus.NOT_EVALUABLE


def test_missing_data_handling_for_unimplemented_models() -> None:
    reg = _registry()
    ev = reg.amc_promotion_evaluations["PDV"]
    assert ev.decision_mapping.has_mapping is False
    assert ev.decision_mapping.missing_data  # never silently empty
    assert "an implementation" in ev.decision_mapping.missing_data


def test_missing_data_handling_for_implemented_models_with_no_amc_hook() -> None:
    reg = _registry()
    ev = reg.amc_promotion_evaluations["GARCH_t"]
    assert ev.decision_mapping.has_mapping is False
    assert ev.decision_mapping.amc_entry_point_citation is None
    assert ev.decision_mapping.missing_data
    assert ev.decision_mapping.reason_not_evaluable is not None
    assert "heimdall/contracts" in ev.decision_mapping.reason_not_evaluable


def test_baseline_comparison_integrity_never_fabricated() -> None:
    reg = _registry()
    for model_id, ev in reg.amc_promotion_evaluations.items():
        static_state = next(e.research_state for e in reg.entries if e.model_id == model_id)
        if static_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
            assert ev.baseline_comparison is None
        else:
            assert ev.baseline_comparison is not None
            assert ev.baseline_comparison.positive_incremental_value_demonstrated is False
            for field in (
                ev.baseline_comparison.incremental_pnl_payoff, ev.baseline_comparison.incremental_drawdown,
                ev.baseline_comparison.incremental_risk, ev.baseline_comparison.incremental_costs,
            ):
                assert field is None, f"{model_id}: no incremental economic metric may be populated"


def test_gate_code_references_point_to_real_files_or_directories() -> None:
    reg = _registry()
    for ev in reg.amc_promotion_evaluations.values():
        for gate in ev.gates:
            if gate.code_reference and "(" not in gate.code_reference:
                candidate = REPO_ROOT / "src" / "model_lab" / gate.code_reference.split("/")[0]
                assert candidate.exists(), f"gate {gate.gate_id} cites a non-existent path: {gate.code_reference}"


def test_leakage_gate_pass_only_for_implemented_models_with_real_remediation_history() -> None:
    reg = _registry()
    for model_id, ev in reg.amc_promotion_evaluations.items():
        defects_gate = next(g for g in ev.gates if g.gate_id == "G9_NO_CRITICAL_DEFECTS")
        entry = next(e for e in reg.entries if e.model_id == model_id)
        if entry.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
            assert defects_gate.state == GateEvidenceState.NOT_APPLICABLE
        else:
            assert defects_gate.state == GateEvidenceState.PASS


def test_nine_gates_per_model_with_correct_cascade_for_implemented_vs_literature() -> None:
    reg = _registry()
    for model_id, ev in reg.amc_promotion_evaluations.items():
        assert len(ev.gates) == 9
        entry = next(e for e in reg.entries if e.model_id == model_id)
        economic_gate = next(g for g in ev.gates if g.gate_id == "G4_ECONOMIC_BASELINE")
        if entry.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
            assert economic_gate.state == GateEvidenceState.NOT_APPLICABLE
        else:
            assert economic_gate.state == GateEvidenceState.FAIL
        assert economic_gate.threshold_status != "INVENTED"


def test_rejected_or_retired_research_state_is_not_conflated_with_amc_evaluated_failed() -> None:
    """regime_2state is REJECTED_OR_RETIRED at the research level (its
    value-add hypothesis was tested and rejected) -- a DIFFERENT finding
    from its AMC evaluation (NOT_EVALUABLE, no decision mapping). Both
    must be visible, never merged into one status."""
    reg = _registry()
    entry = next(e for e in reg.entries if e.model_id == "regime_2state")
    ev = reg.amc_promotion_evaluations["regime_2state"]
    assert entry.research_state == ResearchState.REJECTED_OR_RETIRED
    assert ev.overall_status == EvaluationStatus.NOT_EVALUABLE
    assert any("REJECTED_OR_RETIRED" in note and "NOT_EVALUABLE" in note for note in ev.limitations)


def test_human_authorization_was_never_sought() -> None:
    reg = _registry()
    for ev in reg.amc_promotion_evaluations.values():
        assert ev.human_authorization_state == "NOT_SOUGHT"


def test_full_export_with_amc_evaluations_is_internally_consistent() -> None:
    reg = _registry()
    assert len(reg.amc_promotion_evaluations) == len(reg.entries)
    assert set(reg.amc_promotion_evaluations.keys()) == {e.model_id for e in reg.entries}
    assert reg.research_summary.n_models_amc_not_evaluable == len(reg.entries)


def test_committed_export_file_includes_amc_promotion_evaluations() -> None:
    from model_lab.contracts.model_lab_registry import ModelLabRegistryExport

    export_path = REPO_ROOT / "reports" / "model_lab_registry" / "registry_export.json"
    assert export_path.exists(), "run scripts/build_model_lab_registry.py before this test"
    loaded = ModelLabRegistryExport.model_validate_json(export_path.read_text())
    assert len(loaded.amc_promotion_evaluations) == 26
    for ev in loaded.amc_promotion_evaluations.values():
        assert ev.overall_status == "NOT_EVALUABLE"
