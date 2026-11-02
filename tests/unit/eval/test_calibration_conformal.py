from __future__ import annotations

import numpy as np

from model_lab.eval.calibration import (
    expected_calibration_error,
    interval_coverage,
    pit_uniformity_test,
    reliability_slope,
)
from model_lab.eval.conformal import AciGuard, volatility_normalised_split_conformal


def test_reliability_slope_perfect_calibration() -> None:
    rng = np.random.RandomState(0)
    x = rng.rand(2000)
    y = x + rng.randn(2000) * 0.01
    result = reliability_slope(x, y, n_boot=200)
    assert abs(result.slope - 1.0) < 0.05
    assert result.ci_lower < 1.0 < result.ci_upper


def test_ece_low_for_well_calibrated_forecaster() -> None:
    rng = np.random.RandomState(1)
    probs = rng.rand(5000)
    outcomes = (rng.rand(5000) < probs).astype(float)
    result = expected_calibration_error(probs, outcomes, n_boot=200)
    assert result.ece < 0.05


def test_ece_high_for_badly_calibrated_forecaster() -> None:
    rng = np.random.RandomState(2)
    probs = rng.rand(5000)
    outcomes = (rng.rand(5000) < 0.5).astype(float)  # always 50/50 regardless of stated prob
    result = expected_calibration_error(probs, outcomes, n_boot=200)
    assert result.ece > 0.1


def test_pit_uniformity_accepts_correctly_calibrated_pit() -> None:
    rng = np.random.RandomState(3)
    pit = rng.rand(5000)
    result = pit_uniformity_test(pit)
    assert result.p_value > 0.01


def test_pit_uniformity_rejects_clearly_nonuniform_pit() -> None:
    rng = np.random.RandomState(4)
    pit = rng.beta(2, 5, size=5000)  # concentrated near 0
    result = pit_uniformity_test(pit)
    assert result.p_value < 0.001


def test_interval_coverage_near_nominal() -> None:
    rng = np.random.RandomState(5)
    y = rng.randn(5000)
    lower, upper = np.full(5000, -1.96), np.full(5000, 1.96)
    result = interval_coverage(lower, upper, y, nominal=0.95)
    assert abs(result.empirical_coverage - 0.95) < 0.02


def test_split_conformal_achieves_target_coverage() -> None:
    rng = np.random.RandomState(6)
    n_cal, n_test = 3000, 3000
    sigma_cal = np.abs(rng.randn(n_cal)) + 0.5
    y_cal = rng.randn(n_cal) * sigma_cal
    point_cal = np.zeros(n_cal)
    sigma_test = np.abs(rng.randn(n_test)) + 0.5
    y_test = rng.randn(n_test) * sigma_test
    point_test = np.zeros(n_test)

    result = volatility_normalised_split_conformal(point_cal, y_cal, sigma_cal, point_test, sigma_test, alpha=0.1)
    covered = (y_test >= result.lower) & (y_test <= result.upper)
    assert abs(covered.mean() - 0.9) < 0.03


def test_aci_guard_tracks_nominal_coverage_under_drift() -> None:
    rng = np.random.RandomState(7)
    guard = AciGuard(target_alpha=0.1, gamma=0.05)
    # Simulate a long run where true miscoverage starts at 0.3 then improves
    # to 0.1 -- alpha_t should adapt and not diverge.
    for i in range(2000):
        true_miscoverage = 0.3 if i < 1000 else 0.1
        covered = rng.rand() > true_miscoverage
        guard.update(covered)
    assert 0.0 < guard.alpha_t < 1.0
    assert len(guard.history) == 2000
