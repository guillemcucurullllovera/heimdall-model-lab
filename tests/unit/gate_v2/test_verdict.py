from __future__ import annotations

from model_lab.gate_v2.contracts import Branch, ComponentKind, ComponentResult, Verdict
from model_lab.gate_v2.verdict import (
    assert_closed_list,
    compute_verdict,
    mutation_test_flips_verdict,
)


def _comp(id_: str, gating: bool, kind: ComponentKind, passed: bool | None, missing: bool = False) -> ComponentResult:
    return ComponentResult(
        id=id_, branch=Branch.S, name=id_, statistic="-", evidence="-", rule="-", threshold=None,
        threshold_status="n/a", gating=gating, kind=kind, passed=passed, missing_input=missing,
    )


def test_all_gating_pass_is_sufficient() -> None:
    comps = [
        _comp("A", True, ComponentKind.POWER, True),
        _comp("B", True, ComponentKind.VETO, True),
        _comp("C", False, ComponentKind.REPORT_ONLY, None),
    ]
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.SUFFICIENT
    assert fail_id is None


def test_veto_failure_is_fail() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, True), _comp("B", True, ComponentKind.VETO, False)]
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.FAIL
    assert fail_id == "B"


def test_power_shortfall_without_veto_is_insufficient() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, False), _comp("B", True, ComponentKind.VETO, True)]
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.INSUFFICIENT
    assert fail_id is None


def test_missing_input_on_gating_component_is_fail() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, None, missing=True)]
    verdict, fail_id = compute_verdict(comps)
    assert verdict == Verdict.FAIL
    assert fail_id == "A"


def test_report_only_components_never_affect_verdict() -> None:
    comps_pass = [_comp("A", True, ComponentKind.POWER, True), _comp("R", False, ComponentKind.REPORT_ONLY, False)]
    comps_fail_report = [_comp("A", True, ComponentKind.POWER, True), _comp("R", False, ComponentKind.REPORT_ONLY, True)]
    assert compute_verdict(comps_pass)[0] == compute_verdict(comps_fail_report)[0] == Verdict.SUFFICIENT


def test_mutation_test_flips_verdict_for_every_gating_component() -> None:
    comps = [
        _comp("A", True, ComponentKind.POWER, True),
        _comp("B", True, ComponentKind.POWER, True),
        _comp("C", True, ComponentKind.VETO, True),
    ]
    for c in comps:
        if c.gating:
            assert mutation_test_flips_verdict(comps, c.id), f"flipping {c.id} did not change the verdict"


def test_mutation_test_raises_for_report_only_component() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, True), _comp("R", False, ComponentKind.REPORT_ONLY, False)]
    import pytest

    with pytest.raises(ValueError):
        mutation_test_flips_verdict(comps, "R")


def test_assert_closed_list_passes_for_consistent_declaration() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, True), _comp("R", False, ComponentKind.REPORT_ONLY, None)]
    assert_closed_list(comps, ["A"], ["R"])


def test_assert_closed_list_detects_orphaned_component() -> None:
    comps = [_comp("A", True, ComponentKind.POWER, True), _comp("R", False, ComponentKind.REPORT_ONLY, None)]
    import pytest

    with pytest.raises(AssertionError):
        # declare only A as gating and nothing as report_only -> R is orphaned
        assert_closed_list(comps, ["A"], [])
