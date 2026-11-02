"""Dogfooding control (AT-16): tampered ledger -> QUARANTINED.

This is the same mechanism AT-01 tests directly (ledger/verifier.py's
`model_versions_to_quarantine`); this test frames it explicitly as one of
the six required dogfooding controls, using a slightly richer lineage (a
model with a predecessor) to demonstrate the whole downstream chain gets
quarantined, not just the directly-tampered record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.ledger.storage import LedgerFile
from model_lab.ledger.verifier import verify_chain
from model_lab.ledger.writer import LedgerWriter

pytestmark = pytest.mark.dogfood


def test_tampered_ledger_quarantines_the_whole_downstream_lineage(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    w = LedgerWriter(path, author_id="dogfood")

    w.append(
        LedgerRecordType.MODEL_VERSION,
        {"model_id": "gen0", "model_version_hash": "1" * 64, "lifecycle_state": "RESEARCH", "predecessor_hash": None},
    )
    w.append(
        LedgerRecordType.MODEL_VERSION,
        {"model_id": "gen1", "model_version_hash": "2" * 64, "lifecycle_state": "REPRODUCED", "predecessor_hash": "1" * 64},
    )
    w.append(
        LedgerRecordType.MODEL_VERSION,
        {"model_id": "gen2", "model_version_hash": "3" * 64, "lifecycle_state": "WALK_FORWARD", "predecessor_hash": "2" * 64},
    )

    # Tamper the middle-generation record.
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["body"]["lifecycle_state"] = "PROMOTED"  # attacker tries to forge a promotion
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = LedgerFile(path).read_all()
    result = verify_chain(records)

    assert not result.ok
    assert "gen1" in result.model_versions_to_quarantine
    assert "gen2" in result.model_versions_to_quarantine, "descendant via predecessor_hash must also quarantine"
    assert "gen0" not in result.model_versions_to_quarantine, "the un-tampered ancestor must NOT be falsely flagged"
