"""Integration tests for the full `EmpiricalEvaluationExport` -- runs the
real build end to end (real data, real GARCH/EWMA/regime/anomaly/
changepoint fits, no mocks) and checks the evidence-discipline/no-
fabrication invariants this whole checkpoint exists to enforce.

Slow (several minutes: ~16 real GARCH/EGARCH fits across 6 tickers) --
this is deliberate; a mocked version would not catch a real wiring bug.
"""

from __future__ import annotations

import pytest

from model_lab.command_center_export.empirical_evaluator import build_empirical_evaluation_export
from model_lab.command_center_export.model_registry_data import STATIC_ENTRIES
from model_lab.contracts.model_lab_registry import ResearchState

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def export():
    return build_empirical_evaluation_export()


def test_every_static_entry_is_either_evaluated_or_explained(export) -> None:
    for static in STATIC_ENTRIES:
        assert static.model_id in export.evaluations or static.model_id in export.not_evaluable, (
            f"{static.model_id} is neither evaluated nor marked not_evaluable -- a silently dropped model."
        )


def test_literature_candidates_are_all_not_evaluable_with_a_real_reason(export) -> None:
    literature_ids = {s.model_id for s in STATIC_ENTRIES if s.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE}
    assert literature_ids
    for model_id in literature_ids:
        assert model_id in export.not_evaluable
        assert model_id not in export.evaluations
        assert "no implementation" in export.not_evaluable[model_id].lower()


def test_amc_eligible_is_false_for_every_single_evaluated_model(export) -> None:
    """The one invariant this entire layer must never violate."""
    for model_id, ev in export.evaluations.items():
        assert ev.evidence_discipline.amc_eligible.value is False, f"{model_id} must never be marked AMC-eligible in this increment."
        assert ev.evidence_discipline.amc_eligible.evidence


def test_evidence_discipline_flags_are_not_collapsed(export) -> None:
    """At least one evaluated model must be economically_useful=True while
    shadow_prospective=False (proves these two flags vary independently,
    not a collapsed single status)."""
    found_useful_not_prospective = any(
        ev.evidence_discipline.economically_useful.value and not ev.evidence_discipline.shadow_prospective.value
        for ev in export.evaluations.values()
    )
    assert found_useful_not_prospective

    found_not_useful = any(not ev.evidence_discipline.economically_useful.value for ev in export.evaluations.values())
    assert found_not_useful, "every single model showing 'useful' would itself be an unexamined, suspicious result."


def test_every_evaluated_model_has_implemented_true_with_real_file_evidence(export) -> None:
    for model_id, ev in export.evaluations.items():
        assert ev.evidence_discipline.implemented.value is True
        assert ".py" in ev.evidence_discipline.implemented.evidence, f"{model_id} implemented-evidence does not cite a real file."


def test_vol_targeting_results_never_misattribute_a_different_tickers_data(export) -> None:
    for model_id in ("GARCH_t", "EGARCH_t", "EWMA_0.94", "naive_trailing_rv", "Combo_median_EWMA_GARCH_EGARCH", "regime_2state"):
        ev = export.evaluations[model_id]
        tickers_seen = [r.ticker for r in ev.vol_targeting_results]
        assert len(tickers_seen) == len(set(tickers_seen)), f"{model_id} has duplicate-ticker vol-targeting results."
        for r in ev.vol_targeting_results:
            assert r.model_id == model_id


def test_combo_tail_model_does_not_misattribute_gjr_ts_fz0_comparison(export) -> None:
    """GJR_t_filtered_GPD_FHS and Combo_median_filtered_GPD_FHS must NOT
    carry identical tail_capital_efficiency payloads -- Combo has no
    fz0_mean entry in reports/tail_ladder and must use its own distinct
    (violation-count ablation) evidence instead."""
    gjr = export.evaluations["GJR_t_filtered_GPD_FHS"]
    combo = export.evaluations["Combo_median_filtered_GPD_FHS"]
    assert gjr.tail_capital_efficiency
    assert combo.tail_capital_efficiency == []
    assert "downstream_value_probe" in combo.source_data_refs[0]
    assert combo.evidence_discipline.empirically_evaluated.value is True


def test_anomaly_models_use_the_shared_panel_not_per_model_divergent_data(export) -> None:
    anomaly_ids = ("empirical", "ledoit_wolf", "oas", "min_cov_det", "Isolation_Forest")
    n_flagged_values = set()
    for model_id in anomaly_ids:
        utility = export.evaluations[model_id].risk_reduction_utility
        assert utility is not None
        n_flagged_values.add(utility.n_flagged)
    assert len(n_flagged_values) == 1, "all 5 anomaly models share the same OOS panel/top-decile convention -- n_flagged must match."


def test_changepoint_models_pool_across_all_six_tickers(export) -> None:
    for model_id in ("CUSUM", "BOCPD"):
        utility = export.evaluations[model_id].risk_reduction_utility
        assert utility is not None
        assert utility.n_flagged + utility.n_unflagged > 2000, f"{model_id} utility looks single-ticker, not pooled across 6."


def test_export_is_self_consistent_and_hashed(export) -> None:
    assert export.export_sha256
    assert export.git_commit
    assert len(export.charts) == 7
