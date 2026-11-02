from __future__ import annotations

import numpy as np

from model_lab.gate_v2.branch_p import BranchPEvidence, evaluate_branch_p
from model_lab.gate_v2.verdict import assert_closed_list


def test_missing_freeze_yields_missing_input() -> None:
    ev = BranchPEvidence(freeze_signed=False)
    comps = evaluate_branch_p(ev)
    p4 = next(c for c in comps if c.id == "P4")
    assert p4.passed is False
    assert p4.gating


def test_full_evidence_closed_list() -> None:
    d = np.full(50, 0.2)
    ev = BranchPEvidence(
        freeze_signed=True, freeze_model_version_hash="a" * 64, freeze_bound_b=1.0, d=d, delta=0.05, tier="T1",
        epochs_overlapping=False,
    )
    comps = evaluate_branch_p(ev)
    gating_ids = [c.id for c in comps if c.gating]
    report_ids = [c.id for c in comps if not c.gating]
    assert_closed_list(comps, gating_ids, report_ids)
    assert set(gating_ids) == {"P1", "P2", "P4"}
    assert report_ids == ["P3"]


def test_strong_edge_crosses_threshold() -> None:
    d = np.full(200, 0.3)
    ev = BranchPEvidence(freeze_signed=True, freeze_bound_b=1.0, d=d, delta=0.05, tier="T1", epochs_overlapping=False)
    comps = evaluate_branch_p(ev)
    p1 = next(c for c in comps if c.id == "P1")
    assert p1.passed is True
