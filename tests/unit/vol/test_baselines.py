from __future__ import annotations

import numpy as np

from model_lab.eval.leakage import check_prefix_invariance, future_shuffle_test
from model_lab.vol.baselines import ewma_forecast, ewma_variance, naive_trailing_rv_forecast


def _synthetic_returns(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(6, n) * 0.01


def test_naive_trailing_rv_forecast_no_leakage_prefix_invariance() -> None:
    r = _synthetic_returns()
    result = check_prefix_invariance(lambda x: naive_trailing_rv_forecast(x, horizon=5, window=50), r)
    assert result.all_passed, result


def test_naive_trailing_rv_forecast_no_leakage_future_shuffle() -> None:
    r = _synthetic_returns()
    assert future_shuffle_test(lambda x: naive_trailing_rv_forecast(x, horizon=5, window=50), r, checkpoint=200)


def test_ewma_forecast_no_leakage_prefix_invariance() -> None:
    r = _synthetic_returns()
    result = check_prefix_invariance(lambda x: ewma_forecast(x, horizon=5, lam=0.94), r)
    assert result.all_passed, result


def test_ewma_forecast_no_leakage_future_shuffle() -> None:
    r = _synthetic_returns()
    assert future_shuffle_test(lambda x: ewma_forecast(x, horizon=5, lam=0.94), r, checkpoint=200)


def test_naive_trailing_rv_known_answer() -> None:
    r = np.array([0.01] * 10 + [0.02] * 5)
    out = naive_trailing_rv_forecast(r, horizon=1, window=10)
    # at i=9 (0-indexed, the 10th obs), trailing window is r[0:10], all 0.01 -> mean sq = 0.0001
    assert np.isnan(out[8])
    np.testing.assert_allclose(out[9], 0.0001, rtol=1e-10)


def test_ewma_variance_recursion_known_answer() -> None:
    r = np.array([0.0] * 30 + [0.1])  # seed_window=30 all-zero returns -> seed var = 0, then one big shock
    var = ewma_variance(r, lam=0.9, seed_window=30)
    assert var[29] == 0.0
    expected_next = 0.9 * 0.0 + 0.1 * (0.1**2)
    np.testing.assert_allclose(var[30], expected_next)


def test_ewma_rejects_invalid_lambda() -> None:
    import pytest

    with pytest.raises(ValueError, match="lam"):
        ewma_variance(np.array([0.01, 0.02]), lam=1.5)
