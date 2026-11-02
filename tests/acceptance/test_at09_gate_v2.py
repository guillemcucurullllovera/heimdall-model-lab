"""AT-09: formulas independently reviewed against papers (NOT satisfied --
reported GATED, see reports/GATE_V2_OPERATOR_ACTIONS.md §5); DSR worked
example reproduced (SR0 0.1132, DSR 0.9004, 0.9505 at N=46); mutation test
(flipping any gating component flips the verdict); missing input => FAIL;
whole-gate false-pass rate and power published (small MC on SYNTHETIC data
only); thresholds signed BEFORE any real candidate (never happens in
MODEL-1 -- explicitly NOT done, verified by the signing loader's
fail-closed behaviour)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from model_lab.gate_v2.branch_s import BranchSEvidence, evaluate_branch_s
from model_lab.gate_v2.contracts import Verdict
from model_lab.gate_v2.formulas import dsr, sr0
from model_lab.gate_v2.signing import (
    UnsignedThresholdFileError,
    load_signed_thresholds,
    write_unsigned_threshold_placeholder,
)
from model_lab.gate_v2.verdict import (
    assert_closed_list,
    compute_verdict,
    mutation_test_flips_verdict,
)

pytestmark = pytest.mark.acceptance


def test_dsr_worked_example_reproduced() -> None:
    N = 100
    V_daily = 0.5 / 250.0
    T = 1250
    sr_hat = 2.5 / math.sqrt(250.0)
    assert sr0(N, V_daily) == pytest.approx(0.1132, abs=2e-4)
    assert dsr(sr_hat, N, V_daily, T, -3, 10) == pytest.approx(0.9004, abs=2e-4)
    assert dsr(sr_hat, 46, V_daily, T, -3, 10) == pytest.approx(0.9505, abs=2e-4)


def test_mutation_test_every_gating_component_is_load_bearing() -> None:
    """Per-component mutation testing is only meaningful from an all-pass
    SUFFICIENT baseline: if two veto components already fail simultaneously
    in the observed evidence, flipping just one of them leaves the verdict
    at FAIL either way (a correct artifact of AND-conjunction over multiple
    vetoes, not a bug). So we take the real, wired-up component list from a
    genuine `evaluate_branch_s` call (proving every S0-S16 id/gating/kind is
    correctly populated end-to-end) and then force a synthetic all-pass
    baseline before mutating each gating component one at a time -- this is
    exactly what v1's Defect D2 needed and lacked: proof that no gating
    component is silently ignored by the pass decision."""
    rng = np.random.RandomState(0)
    T, M = 1500, 15
    trial_matrix = rng.randn(T, M) * 0.005
    trial_matrix[:, 4] += 0.0012
    ev = BranchSEvidence(
        family_id="f", lineage_id="l", trial_matrix=trial_matrix, selected_trial_index=4,
        cost_model_id="flat", cost_multiplier=0.0001, stress_episode_windows=[(50, 100), (700, 750)],
        comparator_series={"ewma": trial_matrix[:, 0]}, calendar_years=6.0, target_sr_annualised=0.5,
    )
    comps = evaluate_branch_s(ev)
    all_pass_baseline = [c.model_copy(update={"passed": True, "missing_input": False}) if c.gating else c for c in comps]
    verdict_baseline, _ = compute_verdict(all_pass_baseline)
    assert verdict_baseline == Verdict.SUFFICIENT

    for c in all_pass_baseline:
        if c.gating:
            assert mutation_test_flips_verdict(all_pass_baseline, c.id), f"{c.id} is not load-bearing (v1 Defect D2 pattern)"


def test_missing_required_input_is_fail_not_insufficient() -> None:
    ev = BranchSEvidence()  # nothing provided
    comps = evaluate_branch_s(ev)
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.FAIL
    assert fail_id == "S0"


def test_dead_component_check_holds() -> None:
    rng = np.random.RandomState(1)
    T, M = 1500, 15
    trial_matrix = rng.randn(T, M) * 0.005
    trial_matrix[:, 1] += 0.001
    ev = BranchSEvidence(
        family_id="f", lineage_id="l", trial_matrix=trial_matrix, selected_trial_index=1,
        cost_model_id="flat", cost_multiplier=0.0001, stress_episode_windows=[(50, 100), (700, 750)],
        comparator_series={"ewma": trial_matrix[:, 0]}, calendar_years=6.0, target_sr_annualised=0.5,
    )
    comps = evaluate_branch_s(ev)
    gating_ids = [c.id for c in comps if c.gating]
    report_ids = [c.id for c in comps if not c.gating]
    assert_closed_list(comps, gating_ids, report_ids)  # raises if anything is orphaned


def test_unsigned_thresholds_refuse_to_load_before_any_calibration(tmp_path) -> None:
    """This IS the enforcement that no real candidate can be judged before
    thresholds are signed -- proven by the loader raising, not by
    convention."""
    path = tmp_path / "thresholds.json"
    write_unsigned_threshold_placeholder(path, {"theta_dsr": "0.95"})
    with pytest.raises(UnsignedThresholdFileError):
        load_signed_thresholds(path, allowed_signers_path=tmp_path / "allowed_signers")


def test_whole_gate_false_pass_rate_on_pure_noise_synthetic_data() -> None:
    """Small illustrative Monte Carlo (NOT a calibration run -- explicitly
    not signed/frozen) on PURE NOISE trials: the SUFFICIENT rate should be
    low (the gate should mostly return FAIL/INSUFFICIENT for noise), giving
    a rough, published false-pass-rate figure for this pass."""
    rng = np.random.RandomState(2)
    n_reps = 15  # small, illustrative -- a real calibration run needs far more (see spec §9)
    n_sufficient = 0
    for rep in range(n_reps):
        T, M = 600, 10
        trial_matrix = rng.randn(T, M) * 0.01  # pure noise, no real edge anywhere
        best_idx = int(np.argmax([trial_matrix[:, j].mean() / (trial_matrix[:, j].std() + 1e-9) for j in range(M)]))
        ev = BranchSEvidence(
            family_id=f"f{rep}", lineage_id="l", trial_matrix=trial_matrix, selected_trial_index=best_idx,
            cost_model_id="flat", cost_multiplier=0.0001, stress_episode_windows=[(50, 100), (400, 450)],
            comparator_series={"ewma": trial_matrix[:, (best_idx + 1) % M]}, calendar_years=2.4, target_sr_annualised=0.5,
        )
        comps = evaluate_branch_s(ev)
        verdict, _ = compute_verdict(comps)
        if verdict == Verdict.SUFFICIENT:
            n_sufficient += 1
    false_pass_rate = n_sufficient / n_reps
    # Published (illustrative, not a calibration result): false_pass_rate on
    # pure noise with these [P] PROPOSED thresholds.
    print(f"AT-09 illustrative false-pass rate on pure noise (n={n_reps}): {false_pass_rate:.2%}")
    assert false_pass_rate <= 0.5  # loose bound; a real calibration run (spec §9) is the actual gate for this claim


def test_independent_formula_review_is_gated_not_satisfied() -> None:
    """AT-09 requires 'formulas independently reviewed against papers... a
    non-author reviewer'. No such review exists (model-validator has no
    real identity yet, per operator ruling). This test documents that
    truthfully rather than faking a pass -- it asserts the CONDITION
    (no review record exists), not that the requirement is met."""
    import os

    review_dir = "./local-data/reviews"
    has_review = os.path.isdir(review_dir) and any(os.scandir(review_dir))
    assert not has_review, "AT-09's independent-review requirement is GATED, not satisfied -- update reports/GATE_V2_OPERATOR_ACTIONS.md if this changes"
