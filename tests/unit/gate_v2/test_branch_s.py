from __future__ import annotations

import numpy as np

from model_lab.gate_v2.branch_s import BranchSEvidence, evaluate_branch_s
from model_lab.gate_v2.contracts import Verdict
from model_lab.gate_v2.verdict import assert_closed_list, compute_verdict


def _strong_evidence(seed: int = 0) -> BranchSEvidence:
    rng = np.random.RandomState(seed)
    T, M = 2000, 15
    trial_matrix = rng.randn(T, M) * 0.005
    trial_matrix[:, 2] += 0.0015  # a genuinely, strongly better trial
    return BranchSEvidence(
        family_id="fam1", lineage_id="lin1", trial_matrix=trial_matrix, selected_trial_index=2,
        cost_model_id="flat", cost_multiplier=0.00005,
        stress_episode_windows=[(100, 150), (900, 950), (1500, 1550)],
        comparator_series={"ewma": trial_matrix[:, 0]},
        calendar_years=8.0, target_sr_annualised=0.5,
    )


def test_missing_inputs_yields_all_fail_components() -> None:
    ev = BranchSEvidence()
    comps = evaluate_branch_s(ev)
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.FAIL
    assert fail_id == "S0"


def test_closed_list_holds_for_full_evidence() -> None:
    comps = evaluate_branch_s(_strong_evidence())
    gating_ids = [c.id for c in comps if c.gating]
    report_ids = [c.id for c in comps if not c.gating]
    assert_closed_list(comps, gating_ids, report_ids)
    # exactly S0,S1,S3,S6,S7,S8,S10,S11,S12,S13,S14,S15,S16 gating (13) and
    # S2,S4,S5,S9 report_only (4). S5 (MinBTL) was demoted from gating to
    # report_only 2026-09-30: its "V-BBLZ14" citation was found false by
    # the independent adversarial review (no such theorem/equation exists
    # in the actual PBO paper) and the true primary source is absent from
    # the corpus, so it is no longer treated as verified enough to gate a
    # verdict -- see formulas.py's min_btl docstring and
    # reports/MODEL1_REVIEW_REMEDIATION.md P1-2.
    assert set(gating_ids) == {"S0", "S1", "S3", "S6", "S7", "S8", "S10", "S11", "S12", "S13", "S14", "S15", "S16"}
    assert set(report_ids) == {"S2", "S4", "S5", "S9"}


def test_verdict_is_one_of_the_three_tokens() -> None:
    comps = evaluate_branch_s(_strong_evidence())
    verdict, _ = compute_verdict(comps)
    assert verdict in (Verdict.SUFFICIENT, Verdict.INSUFFICIENT, Verdict.FAIL)


def test_component_count_is_17() -> None:
    comps = evaluate_branch_s(_strong_evidence())
    assert len(comps) == 17  # S0..S16
