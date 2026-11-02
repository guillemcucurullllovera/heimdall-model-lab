"""Real-data smoke tests for the two new model-native diagnostic charts.
Not mocked -- these call the same already-tested GARCH/EWMA functions the
vol ladder itself uses, on the real frozen dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.economic_evaluation.diagnostics import (
    build_calibration_chart,
    build_predicted_vs_realized_chart,
)

pytestmark = pytest.mark.slow

REPORTS_DIR = Path("reports")


def test_predicted_vs_realized_chart_has_six_series_including_realized_comparator() -> None:
    registry = DatasetSnapshotRegistry.load_from_directory()
    chart = build_predicted_vs_realized_chart(registry, "TLT")
    assert len(chart.series) == 6
    labels = {s.label for s in chart.series}
    assert "realized_variance" in labels
    comparator_series = [s for s in chart.series if s.label == "realized_variance"][0]
    assert comparator_series.role == "comparator"
    assert chart.n is not None and chart.n > 0
    assert all(len(s.points) > 0 for s in chart.series)


def test_predicted_vs_realized_chart_points_are_monthly_not_daily() -> None:
    registry = DatasetSnapshotRegistry.load_from_directory()
    chart = build_predicted_vs_realized_chart(registry, "SPY")
    # ~8 years of OOS data downsampled to monthly means should be on the order of
    # dozens, not thousands, of points -- a sanity bound against accidental daily output.
    assert 12 < len(chart.series[0].points) < 150


def test_calibration_chart_reuses_committed_mz_slope_exactly() -> None:
    import json

    chart = build_calibration_chart(REPORTS_DIR, ("SPY", "QQQ", "TLT", "IEF", "GLD", "USO"))
    assert len(chart.series) == 5
    spy_ladder = json.loads((REPORTS_DIR / "vol_ladder" / "SPY_h1.json").read_text())
    garch_series = [s for s in chart.series if s.label == "GARCH_t"][0]
    spy_point = [p for p in garch_series.points if p.x_label == "SPY"][0]
    assert float(spy_point.value) == pytest.approx(spy_ladder["results"]["GARCH_t"]["mz_slope"])


def test_both_charts_have_distinct_self_referential_hashes() -> None:
    registry = DatasetSnapshotRegistry.load_from_directory()
    chart_a = build_predicted_vs_realized_chart(registry, "SPY")
    chart_b = build_calibration_chart(REPORTS_DIR, ("SPY", "QQQ", "TLT", "IEF", "GLD", "USO"))
    assert chart_a.chart_sha256 != chart_b.chart_sha256
