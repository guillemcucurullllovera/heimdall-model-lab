from __future__ import annotations

import numpy as np

from model_lab.economic_evaluation.risk_reduction_utility import (
    anomaly_risk_reduction_utility,
    changepoint_risk_reduction_utility,
)


def test_anomaly_utility_none_when_too_few_valid_observations() -> None:
    n = 20
    scores = np.full(n, np.nan)
    panel_returns = np.zeros(n)
    result = anomaly_risk_reduction_utility("test", scores, panel_returns, list(range(n)))
    assert result is None


def test_anomaly_utility_detects_real_risk_avoidance_value() -> None:
    rng = np.random.default_rng(0)
    n = 300
    scores = rng.normal(0, 1, n)
    panel_returns = rng.normal(0.0005, 0.005, n)
    # Make the top-decile-scored days' NEXT day return clearly worse.
    oos = list(range(n - 1))
    threshold = np.percentile(scores[np.array(oos)], 90)
    flagged_idx = [i for i in oos if scores[i] >= threshold]
    for i in flagged_idx:
        panel_returns[i + 1] = -0.05

    result = anomaly_risk_reduction_utility("test", scores, panel_returns, oos)
    assert result is not None
    assert float(result.mean_forward_return_flagged_days) < float(result.mean_forward_return_unflagged_days)
    assert "risk-avoidance use" in result.interpretation
    assert float(result.hypothetical_avoided_loss_if_flat_on_flagged_days) < 0


def test_anomaly_utility_honestly_reports_no_value_when_none_exists() -> None:
    rng = np.random.default_rng(1)
    n = 300
    scores = rng.normal(0, 1, n)
    panel_returns = rng.normal(0.0005, 0.005, n)  # independent of scores -- no real signal
    oos = list(range(n - 1))
    result = anomaly_risk_reduction_utility("test", scores, panel_returns, oos)
    assert result is not None
    assert "no risk-avoidance value demonstrated" in result.interpretation.lower() or "worse on average" in result.interpretation


def test_anomaly_utility_none_when_insufficient_flagged_days() -> None:
    n = 40
    scores = np.zeros(n)  # identical scores -> percentile threshold flags ~0 or all, never a clean 5/5 split
    panel_returns = np.zeros(n)
    result = anomaly_risk_reduction_utility("test", scores, panel_returns, list(range(n - 1)))
    assert result is None


def test_changepoint_utility_boolean_alarm_path() -> None:
    rng = np.random.default_rng(2)
    n = 300
    alarm = np.zeros(n, dtype=bool)
    log_returns = rng.normal(0.0003, 0.01, n)
    oos = list(range(n - 1))
    # Flag 20 days and make their next-day return clearly worse.
    flagged_positions = oos[::15][:20]
    alarm[flagged_positions] = True
    for i in flagged_positions:
        log_returns[i + 1] = -0.03

    result = changepoint_risk_reduction_utility("CUSUM", alarm, log_returns, oos, is_probability=False)
    assert result is not None
    assert result.n_flagged == 20
    assert float(result.mean_forward_return_flagged_days) < float(result.mean_forward_return_unflagged_days)


def test_changepoint_utility_probability_path_uses_top_decile() -> None:
    rng = np.random.default_rng(3)
    n = 300
    prob = rng.uniform(0, 1, n)
    log_returns = rng.normal(0.0003, 0.01, n)
    oos = list(range(n - 1))
    result = changepoint_risk_reduction_utility("BOCPD", prob, log_returns, oos, is_probability=True)
    assert result is not None
    # np.percentile(..., 90) on uniform draws flags ~10% of observations; bound it rather
    # than assert exact equality (tie-handling can shift the count by a few).
    assert abs(result.n_flagged - round(len(oos) * 0.10)) <= 5


def test_changepoint_utility_none_when_too_few_valid_observations() -> None:
    n = 20
    alarm = np.zeros(n, dtype=bool)
    log_returns = np.zeros(n)
    result = changepoint_risk_reduction_utility("CUSUM", alarm, log_returns, list(range(n)), is_probability=False)
    assert result is None
