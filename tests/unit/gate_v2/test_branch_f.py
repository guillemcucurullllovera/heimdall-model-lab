from __future__ import annotations

import numpy as np

from model_lab.gate_v2.branch_f import BranchFEvidence, evaluate_branch_f
from model_lab.gate_v2.verdict import assert_closed_list


def test_missing_inputs_reports_missing_not_crash() -> None:
    ev = BranchFEvidence()
    comps = evaluate_branch_f(ev)
    assert all(c.missing_input for c in comps if c.id in {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"})


def test_closed_list_holds_for_full_evidence() -> None:
    rng = np.random.RandomState(0)
    n = 1000
    loss_incumbent = np.abs(rng.randn(n)) + 0.5
    loss_challenger = np.abs(rng.randn(n)) * 0.3
    loss_naive = np.abs(rng.randn(n)) + 1.0
    era = np.repeat(np.arange(4), n // 4)
    prob = rng.rand(n)
    outcome = (rng.rand(n) < prob).astype(float)
    dec_inc = (rng.rand(n) > 0.5).astype(int)
    dec_chal = np.where(rng.rand(n) < 0.2, 1 - dec_inc, dec_inc)

    ev = BranchFEvidence(
        loss_incumbent=loss_incumbent, loss_challenger=loss_challenger, loss_naive=loss_naive, is_nested=False,
        cpcv_path_losses=[(loss_incumbent[i::5], loss_challenger[i::5]) for i in range(5)],
        era_labels=era, forecast_prob=prob, outcome=outcome, decision_incumbent=dec_inc, decision_challenger=dec_chal,
        guard_values={"cvar": (0.02, 0.01)}, h=1,
    )
    comps = evaluate_branch_f(ev)
    gating_ids = [c.id for c in comps if c.gating]
    report_ids = [c.id for c in comps if not c.gating]
    assert_closed_list(comps, gating_ids, report_ids)
    assert set(gating_ids) == {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"}
    assert report_ids == []
    assert len(comps) == 9
