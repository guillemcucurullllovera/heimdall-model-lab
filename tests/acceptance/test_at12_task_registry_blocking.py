"""AT-12: a task with a BLOCKED dataset cannot accept a prereg; status
derives mechanically."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.contracts.task import TaskStatus
from model_lab.governance.prereg_gate import PreregRejected, make_prereg_gate
from model_lab.ledger.writer import LedgerWriter
from model_lab.tasks import build_default_task_registry

pytestmark = pytest.mark.acceptance


def test_prereg_against_blocked_task_is_rejected_by_the_ledger(tmp_path: Path) -> None:
    registry = build_default_task_registry()
    assert registry["T-VOL"].status == TaskStatus.BLOCKED

    def status_of(task_id: str) -> TaskStatus:
        return TaskStatus(registry[task_id].status)

    writer = LedgerWriter(tmp_path / "ledger.ndjson", author_id="test-author")
    gate = make_prereg_gate(status_of)

    with pytest.raises(PreregRejected):
        writer.append(
            LedgerRecordType.PREREG,
            {"prereg_id": "pr-1", "task_ids": ["T-VOL"], "hypothesis": "GJR-t beats EWMA"},
            pre_append_check=gate,
        )

    # And nothing was actually appended — the check runs before the write.
    from model_lab.ledger.storage import LedgerFile

    assert LedgerFile(tmp_path / "ledger.ndjson").read_all() == []


def test_prereg_against_a_ready_task_is_accepted(tmp_path: Path) -> None:
    registry = build_default_task_registry()

    def status_of(task_id: str) -> TaskStatus:
        if task_id == "T-SYNTHETIC-READY":
            return TaskStatus.DATA_READY
        return TaskStatus(registry[task_id].status)

    writer = LedgerWriter(tmp_path / "ledger.ndjson", author_id="test-author")
    gate = make_prereg_gate(status_of)

    record = writer.append(
        LedgerRecordType.PREREG,
        {"prereg_id": "pr-2", "task_ids": ["T-SYNTHETIC-READY"], "hypothesis": "a dogfooding control"},
        pre_append_check=gate,
    )
    assert record.seq == 0


def test_status_is_derived_not_discretionary() -> None:
    # Post-remediation (P0-1/P0-4): T-TAIL is the one task the mechanical
    # recompute now shows DATA_READY, backed by real registry-resolved
    # DatasetSnapshots -- see tests/unit/tasks/test_registry.py. Every other
    # task remains BLOCKED. This is still asserted structurally (via the
    # contract's own validator, not by this test re-deriving it) simply by
    # successfully constructing the registry at all: Task construction
    # raises if `status` doesn't match `derive_task_status(data_requirements)`.
    registry = build_default_task_registry()
    for task_id, task in registry.items():
        if task_id == "T-TAIL":
            assert task.status == TaskStatus.DATA_READY
        else:
            assert task.status == TaskStatus.BLOCKED, task_id


def test_prereg_against_the_now_unblocked_t_tail_is_accepted(tmp_path: Path) -> None:
    """Real end-to-end confirmation that T-TAIL's mechanical unblock (P0-1)
    actually flows through to the ledger's own prereg gate, not just the
    Task object in isolation."""
    registry = build_default_task_registry()
    assert registry["T-TAIL"].status == TaskStatus.DATA_READY

    def status_of(task_id: str) -> TaskStatus:
        return TaskStatus(registry[task_id].status)

    writer = LedgerWriter(tmp_path / "ledger.ndjson", author_id="test-author")
    gate = make_prereg_gate(status_of)

    record = writer.append(
        LedgerRecordType.PREREG,
        {"prereg_id": "pr-3", "task_ids": ["T-TAIL"], "hypothesis": "GJR-t-filtered GPD/FHS beats unconditional GPD-POT"},
        pre_append_check=gate,
    )
    assert record.seq == 0
