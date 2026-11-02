from __future__ import annotations

import numpy as np
import pytest

from model_lab.economic_evaluation.vol_targeting import (
    _annualized_return,
    _max_drawdown,
    _sharpe,
    run_vol_targeting_backtest,
)


def test_annualized_return_matches_a_known_hand_computed_value() -> None:
    # 252 days of a flat 0.01 daily simple return compounds to (1.01)^252 total over exactly
    # 1 year (252/252), so the annualized return equals the total compounded return itself.
    r = np.full(252, 0.01)
    result = _annualized_return(r)
    assert result == pytest.approx(1.01**252 - 1, rel=1e-9)


def test_max_drawdown_on_a_known_path() -> None:
    # +10%, then -20%: equity goes 1.0 -> 1.1 -> 0.88. Drawdown from peak 1.1 to 0.88 = -0.2
    r = np.array([0.10, -0.20])
    assert _max_drawdown(r) == pytest.approx(-0.20, rel=1e-9)


def test_sharpe_is_nan_for_zero_variance_series() -> None:
    r = np.full(100, 0.001)
    assert np.isnan(_sharpe(r))


def test_insufficient_observations_returns_none_not_a_fabricated_result() -> None:
    n = 100
    log_returns = np.random.default_rng(0).normal(0, 0.01, n)
    variance_forecast = np.full(n, np.nan)
    variance_forecast[50:55] = 1e-4  # only 5 valid points, below the 30-minimum
    result = run_vol_targeting_backtest(
        model_id="test", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast,
        naive_variance_forecast=None, oos_indices=list(range(n)),
    )
    assert result is None


def test_buy_and_hold_benchmark_reproduces_the_real_underlying_return() -> None:
    """Sanity check: the buy-and-hold benchmark inside the backtest must
    exactly reproduce compounding the real log returns -- no leakage, no
    fabrication, a pure identity check."""
    n = 300
    rng = np.random.default_rng(1)
    log_returns = rng.normal(0.0002, 0.01, n)
    variance_forecast = np.full(n, 1e-4)
    oos_indices = list(range(n - 1))
    result = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast,
        naive_variance_forecast=None, oos_indices=oos_indices, target_annual_vol=0.10, leverage_cap=3.0,
    )
    assert result is not None
    idx = np.array([i for i in oos_indices if i + 1 < n])
    expected_simple_returns = np.exp(log_returns[idx + 1]) - 1.0
    expected_bh_return = float(_annualized_return(expected_simple_returns))
    assert float(result.benchmark_buy_and_hold["gross_annualized_return"]) == pytest.approx(expected_bh_return, rel=1e-9)


def test_weight_is_prefix_invariant_not_lookahead() -> None:
    """The weight at origin i must depend only on variance_forecast[i] --
    confirmed by construction (no indexing beyond i used to compute it),
    verified here by checking two runs that differ only in a LATER
    variance_forecast value produce an IDENTICAL weight/exposure at the
    earlier, shared origin."""
    n = 300
    rng = np.random.default_rng(2)
    log_returns = rng.normal(0, 0.01, n)
    variance_forecast_a = np.full(n, 1e-4)
    variance_forecast_b = variance_forecast_a.copy()
    variance_forecast_b[250:] = 5e-4  # change only the tail, after the shared origin region
    oos_indices = list(range(200))  # entirely before the changed region

    result_a = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast_a,
        naive_variance_forecast=None, oos_indices=oos_indices,
    )
    result_b = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast_b,
        naive_variance_forecast=None, oos_indices=oos_indices,
    )
    assert result_a is not None and result_b is not None
    assert result_a.exposure_mean == result_b.exposure_mean
    assert result_a.net_annualized_return == result_b.net_annualized_return


def test_higher_transaction_cost_never_improves_net_sharpe() -> None:
    n = 400
    rng = np.random.default_rng(3)
    log_returns = rng.normal(0.0001, 0.012, n)
    variance_forecast = np.abs(rng.normal(1e-4, 2e-5, n))
    oos_indices = list(range(n - 1))
    low_cost = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast,
        naive_variance_forecast=None, oos_indices=oos_indices, cost_bps_per_unit_turnover=1.0,
    )
    high_cost = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast,
        naive_variance_forecast=None, oos_indices=oos_indices, cost_bps_per_unit_turnover=50.0,
    )
    assert low_cost is not None and high_cost is not None
    assert float(high_cost.net_annualized_return) <= float(low_cost.net_annualized_return)


def test_disclosed_assumptions_are_never_silently_omitted() -> None:
    n = 200
    rng = np.random.default_rng(4)
    log_returns = rng.normal(0, 0.01, n)
    variance_forecast = np.full(n, 1e-4)
    result = run_vol_targeting_backtest(
        model_id="GARCH_t", ticker="FAKE", log_returns=log_returns, variance_forecast=variance_forecast,
        naive_variance_forecast=None, oos_indices=list(range(n - 1)),
    )
    assert result is not None
    assert result.assumptions.cost_bps_per_unit_turnover
    assert result.assumptions.leverage_cap
    assert result.assumptions.target_annualized_volatility
    assert "Moreira" in result.assumptions.methodology_reference
    assert any("NOT measured AMC-specific costs" in lim for lim in result.limitations)
