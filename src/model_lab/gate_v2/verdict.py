"""Gate v2 closed-list conjunction verdict rule — §3 / §5 of the verified
research extraction.

"Conjunction rule: verdict = AND over the closed list of gating components
of the branch. Every component the code computes is either gating or
explicitly report_only with a stated reason; the config hash covers both
lists; a unit test fails if the code computes any statistic that is in
neither list."

Verdict logic (verbatim structure):
1. Any missing/invalid required input, or a VETO-kind gating component
   failing -> FAIL immediately (fail-closed; "missing_input_policy" is
   always "FAIL" per-component).
2. Else if every gating component passes -> SUFFICIENT.
3. Else (only POWER/ADEQUACY-kind components short of their bar, no veto
   tripped, inputs otherwise valid) -> INSUFFICIENT.

"INSUFFICIENT never rejects and never advances; it waits for data."
"""

from __future__ import annotations

from model_lab.gate_v2.contracts import ComponentKind, ComponentResult, Verdict


def compute_verdict(components: list[ComponentResult]) -> tuple[Verdict, str | None]:
    """Returns (verdict, first_failing_veto_id). Pure function over the
    already-computed component list -- computing the components themselves
    (running the statistics) is the branch modules' job; this function only
    implements the conjunction rule."""
    gating = [c for c in components if c.gating]

    for c in gating:
        if c.missing_input or c.passed is None:
            return Verdict.FAIL, c.id
        if c.kind == ComponentKind.VETO and not c.passed:
            return Verdict.FAIL, c.id

    if all(c.passed for c in gating):
        return Verdict.SUFFICIENT, None

    return Verdict.INSUFFICIENT, None


def assert_closed_list(components: list[ComponentResult], declared_gating_ids: list[str], declared_report_only_ids: list[str]) -> None:
    """The 'dead-component test': every statistic the code computes must
    appear in gating UNION report_only, nothing orphaned (v1's Defect D2:
    PBO/reality_check_p/bhy_haircut were computed and returned but silently
    ignored by the pass decision)."""
    computed_ids = {c.id for c in components}
    gating_ids = {c.id for c in components if c.gating}
    report_only_ids = {c.id for c in components if not c.gating}

    if gating_ids != set(declared_gating_ids):
        raise AssertionError(f"gating id mismatch: computed={gating_ids}, declared={set(declared_gating_ids)}")
    if report_only_ids != set(declared_report_only_ids):
        raise AssertionError(f"report_only id mismatch: computed={report_only_ids}, declared={set(declared_report_only_ids)}")
    if computed_ids != gating_ids | report_only_ids:
        orphaned = computed_ids - (gating_ids | report_only_ids)
        raise AssertionError(f"orphaned component(s) in neither list: {orphaned}")


def mutation_test_flips_verdict(components: list[ComponentResult], component_id: str) -> bool:
    """Flip a single gating component's `passed` and confirm the verdict
    changes. Returns True if the flip changed the verdict (the expected,
    correct behaviour for every gating component); used by
    tests/acceptance/test_at09_gate_v2.py to prove every gating component
    is load-bearing (not silently ignored, per v1's Defect D2)."""
    target = next((c for c in components if c.id == component_id), None)
    if target is None or not target.gating:
        raise ValueError(f"{component_id!r} is not a gating component of this list")
    verdict_before, _ = compute_verdict(components)
    flipped = [c.model_copy(update={"passed": not c.passed}) if c.id == component_id else c for c in components]
    verdict_after, _ = compute_verdict(flipped)
    return verdict_before != verdict_after
