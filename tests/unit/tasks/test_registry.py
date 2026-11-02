from __future__ import annotations

from model_lab.contracts.task import PromotionClass, TaskStatus
from model_lab.tasks import build_default_task_registry

# Updated post-remediation (MODEL1_ADVERSARIAL_REVIEW.md / P0-1, P0-4): T-TAIL
# is the one task the mechanical recompute now shows DATA_READY/RESEARCH_ONLY,
# backed by real, registry-resolved DatasetSnapshots (including the USO
# split-adjustment fix) -- see C1_C5_UPDATED_READINESS.md. Every other task
# remains BLOCKED for documented, still-true reasons.
STILL_BLOCKED_TASKS = {"T-VOL", "T-STATE", "T-DOWNSTREAM", "T-ANOM", "T-CHANGE", "T-CAL", "T-VRP"}


def test_all_eight_tasks_present() -> None:
    reg = build_default_task_registry()
    assert set(reg) == {
        "T-VOL",
        "T-TAIL",
        "T-STATE",
        "T-DOWNSTREAM",
        "T-ANOM",
        "T-CHANGE",
        "T-CAL",
        "T-VRP",
    }


def test_still_blocked_tasks_remain_blocked() -> None:
    reg = build_default_task_registry()
    for task_id in STILL_BLOCKED_TASKS:
        task = reg[task_id]
        assert task.status == TaskStatus.BLOCKED, f"{task_id} expected BLOCKED"
        assert task.promotion_class == PromotionClass.BLOCKED


def test_t_tail_is_research_only_backed_by_real_registry_resolution() -> None:
    """T-TAIL's DataRequirements must be registry-resolved (dataset_id set,
    research_state matching a real DatasetSnapshot) -- not hand-typed. This
    is a structural check: if any requirement still has dataset_id=None, the
    'unblocked by real data' claim would be false."""
    reg = build_default_task_registry()
    t_tail = reg["T-TAIL"]
    assert t_tail.status == TaskStatus.DATA_READY
    assert t_tail.promotion_class == PromotionClass.RESEARCH_ONLY
    for req in t_tail.data_requirements:
        assert req.dataset_id is not None, f"{req.dataset_role} has no resolved dataset_id"
        assert req.research_state == "research_only_unlicensed"


def test_every_blocked_task_has_at_least_one_blocked_requirement_documented() -> None:
    reg = build_default_task_registry()
    for task_id in STILL_BLOCKED_TASKS:
        task = reg[task_id]
        blocked_reqs = [r for r in task.data_requirements if r.status == "BLOCKED"]
        assert blocked_reqs, task_id
        for r in blocked_reqs:
            assert r.blockers, f"{task_id}/{r.dataset_role} has no documented blocker"
