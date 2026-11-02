from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.ledger.storage import CorruptLedgerError, LedgerFile
from model_lab.ledger.writer import LedgerWriter


def test_read_all_on_missing_file_returns_empty() -> None:
    lf = LedgerFile(Path("/tmp/definitely-does-not-exist-model-lab-test.ndjson"))
    assert lf.read_all() == []


def test_last_record_none_when_empty(tmp_path: Path) -> None:
    lf = LedgerFile(tmp_path / "l.ndjson")
    assert lf.last_record() is None


def test_append_then_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "l.ndjson"
    w = LedgerWriter(path, author_id="a")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "h1"})
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "h2"})

    lf = LedgerFile(path)
    records = lf.read_all()
    assert len(records) == 2
    assert records[0].body == {"text": "h1"}
    assert records[1].seq == 1
    assert lf.last_record().seq == 1


def test_non_tail_corruption_raises_corrupt_ledger_error(tmp_path: Path) -> None:
    path = tmp_path / "l.ndjson"
    w = LedgerWriter(path, author_id="a")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "h1"})
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "h2"})

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "not valid json {{"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CorruptLedgerError):
        LedgerFile(path).read_all()


def test_auto_repair_drops_torn_tail_on_read(tmp_path: Path) -> None:
    path = tmp_path / "l.ndjson"
    w = LedgerWriter(path, author_id="a")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "h1"})

    with open(path, "ab") as f:
        f.write(b'{"seq": 1, "incomplete')

    records = LedgerFile(path).read_all(auto_repair=True)
    assert len(records) == 1
    # File on disk was physically truncated by auto_repair.
    assert path.read_text(encoding="utf-8").count("\n") == 1
