from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.replay.observer import build_input_bundle, list_cycle_files, list_known_cycle_ids


@pytest.fixture
def fake_amc_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "amc4"
    cycles = root / "cycles"
    cycles.mkdir(parents=True)
    journal = root / "amc4_journal.ndjson"

    (cycles / "cycle-001-20260101_candidates.json").write_text("[]", encoding="utf-8")
    (cycles / "cycle-002-20260101_candidates.json").write_text('[{"x": 1}]', encoding="utf-8")
    (cycles / "cycle-002-20260101_problem.json").write_text('{"p": 1}', encoding="utf-8")

    journal_records = [
        {"record_type": "amc4_source_health_snapshot", "experiment_id": "exp-1", "cycle_id": "cycle-002-20260101", "sources": []},
        {"record_type": "amc4_cycle_created", "experiment_id": "exp-1", "cycle_id": "cycle-002-20260101"},
    ]
    journal.write_text("\n".join(json.dumps(r) for r in journal_records) + "\n", encoding="utf-8")

    monkeypatch.setattr("model_lab.replay.observer.AMC_ROOT", root)
    monkeypatch.setattr("model_lab.replay.observer.CYCLES_DIR", cycles)
    monkeypatch.setattr("model_lab.replay.observer.JOURNAL_PATH", journal)
    return root


def test_list_known_cycle_ids(fake_amc_root: Path) -> None:
    ids = list_known_cycle_ids()
    assert ids == ["cycle-001-20260101", "cycle-002-20260101"]


def test_list_cycle_files_finds_all_files_for_a_cycle(fake_amc_root: Path) -> None:
    files = list_cycle_files("cycle-002-20260101")
    names = sorted(p.name for p in files)
    assert names == ["cycle-002-20260101_candidates.json", "cycle-002-20260101_problem.json"]


def test_list_cycle_files_empty_for_unknown_cycle(fake_amc_root: Path) -> None:
    assert list_cycle_files("cycle-999-20260101") == []


def test_build_input_bundle_hashes_real_files_and_journal(fake_amc_root: Path) -> None:
    bundle = build_input_bundle("cycle-002-20260101", experiment_id="exp-1")
    assert bundle.cycle_id == "cycle-002-20260101"
    assert bundle.experiment_id == "exp-1"
    assert len(bundle.candidate_file_hashes) == 2  # candidates.json + problem.json
    assert bundle.source_health_snapshot_hash is not None
    assert bundle.journal_prefix_byte_length > 0


def test_build_input_bundle_no_source_health_snapshot_for_unknown_experiment(fake_amc_root: Path) -> None:
    bundle = build_input_bundle("cycle-002-20260101", experiment_id="no-such-experiment")
    assert bundle.source_health_snapshot_hash is None
