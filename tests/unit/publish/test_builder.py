from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from model_lab.contracts.chart_spec import (
    Causality,
    ChartSeries,
    ChartSeriesPoint,
    ChartSpec,
    DataAuthority,
)
from model_lab.contracts.evidence_report import EvidenceReport, ScoreResult
from model_lab.publish.builder import build_publication, publication_dir
from model_lab.publish.verify import verify_publication

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _evidence_report() -> EvidenceReport:
    return EvidenceReport(
        report_id="er-1", model_version_hash="a" * 64, task_id="T-TAIL", prereg_id="pr-1", run_id="run-1",
        as_of_utc=NOW, primary_score_result=ScoreResult(name="FZ0", value="0.01", n=100, n_independent=10.0),
        report_sha256="b" * 64,
    )


def _chart() -> ChartSpec:
    return ChartSpec(
        chart_id="c1", decision_question="q?", incumbent_id="incumbent",
        series=[ChartSeries(label="incumbent", role="incumbent", points=[ChartSeriesPoint(x_label="t0", value="1")])],
        causality=Causality.NOT_APPLICABLE, data_authority=DataAuthority.PROVISIONAL, chart_sha256="c" * 64,
    )


def test_publication_dir_path_convention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    d = publication_dir("m1", "a" * 64, "2026-09-30")
    assert d == tmp_path / "m1" / ("a" * 64) / "2026-09-30"


def test_build_publication_writes_all_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    manifest = build_publication(
        model_id="m1", model_version_hash="a" * 64, as_of="2026-09-30", ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(), charts=[_chart()], model_card={"x": 1}, datasheet={"y": 2},
    )
    out_dir = tmp_path / "m1" / ("a" * 64) / "2026-09-30"
    assert (out_dir / "evidence.json").exists()
    assert (out_dir / "charts" / "c1.json").exists()
    assert (out_dir / "model_card.json").exists()
    assert (out_dir / "datasheet.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert set(manifest.file_hashes) == {"evidence.json", "charts/c1.json", "model_card.json", "datasheet.json"}


def test_build_publication_is_immutable_content_hashes_are_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from model_lab.canon.jcs import sha256_bytes

    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    manifest = build_publication(
        model_id="m1", model_version_hash="a" * 64, as_of="2026-09-30", ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(), charts=[_chart()], model_card={"x": 1}, datasheet={"y": 2},
    )
    out_dir = tmp_path / "m1" / ("a" * 64) / "2026-09-30"
    for rel_path, expected_hash in manifest.file_hashes.items():
        assert sha256_bytes((out_dir / rel_path).read_bytes()) == expected_hash


def test_verify_publication_ok_on_untampered_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    build_publication(
        model_id="m1", model_version_hash="a" * 64, as_of="2026-09-30", ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(), charts=[_chart()], model_card={"x": 1}, datasheet={"y": 2},
    )
    out_dir = tmp_path / "m1" / ("a" * 64) / "2026-09-30"
    result = verify_publication(out_dir)
    assert result.ok
    assert not result.mismatched_files
    assert not result.missing_files
