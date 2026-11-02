"""AT-12 / §8, §10: "a prereg whose... a prereg against a BLOCKED task is
rejected by the ledger." Implemented as a `PreAppendCheck` (see
ledger/writer.py) so the rejection is atomic with the append attempt — a
prereg can never sneak into the ledger via a race between a status check
and the write.
"""

from __future__ import annotations

from collections.abc import Callable

from model_lab.contracts.ledger_record import LedgerRecord, LedgerRecordType
from model_lab.contracts.task import TaskStatus, accepts_prereg

TaskStatusLookup = Callable[[str], TaskStatus]


class PreregRejected(Exception):
    pass


def make_prereg_gate(task_status_lookup: TaskStatusLookup):
    """Returns a `PreAppendCheck` bound to `task_status_lookup` (typically
    `TaskRegistry.status_of`), for use as
    `LedgerWriter.append(..., pre_append_check=make_prereg_gate(registry.status_of))`.
    """

    def check(record: LedgerRecord, existing: list[LedgerRecord]) -> None:
        if record.record_type != LedgerRecordType.PREREG:
            return
        task_ids = record.body.get("task_ids", [])
        for task_id in task_ids:
            status = task_status_lookup(task_id)
            if not accepts_prereg(status):
                raise PreregRejected(
                    f"task {task_id!r} has status {status!r}; the ledger refuses a prereg "
                    "against a BLOCKED task (§8/§10, AT-12)"
                )

    return check
