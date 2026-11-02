from __future__ import annotations

import numpy as np

from model_lab.eval.tests_dm import clark_west, diebold_mariano, dm_hac_lag, giacomini_white


def test_dm_hac_lag_formula() -> None:
    assert dm_hac_lag(h=1, n=1000) == 1 - 1 + 10  # ceil(1000^(1/3)) == 10
    assert dm_hac_lag(h=5, n=1000) == 5 - 1 + 10


def test_dm_detects_a_clearly_better_model() -> None:
    rng = np.random.RandomState(0)
    n = 2000
    loss_a = np.abs(rng.randn(n)) + 1.0  # bad model
    loss_b = np.abs(rng.randn(n)) * 0.3  # clearly better model
    result = diebold_mariano(loss_a, loss_b, h=1)
    assert result.mean_loss_diff > 0
    assert result.hln_p_value < 0.01


def test_dm_null_case_size_near_nominal() -> None:
    # Under H0 (loss_a and loss_b i.i.d. identical distribution, no true
    # difference), the HLN-corrected DM test should reject at roughly the
    # nominal rate across many independent trials -- not aiming for an exact
    # bound here (that's the readiness-pack's own R4 simulation), just
    # sanity that it's in a plausible range and not wildly miscalibrated.
    rng = np.random.RandomState(1)
    n_trials = 300
    rejections = 0
    for _ in range(n_trials):
        loss_a = rng.randn(500) ** 2
        loss_b = rng.randn(500) ** 2
        r = diebold_mariano(loss_a, loss_b, h=1)
        if r.hln_p_value < 0.05:
            rejections += 1
    rate = rejections / n_trials
    assert 0.0 <= rate <= 0.15  # loose bound; classic DM over-rejects up to ~14% per R4


def test_clark_west_favours_unrestricted_when_it_is_truly_better() -> None:
    rng = np.random.RandomState(2)
    n = 2000
    y = rng.randn(n)
    forecast_restricted = np.zeros(n)  # always predicts 0 (misses the signal)
    forecast_unrestricted = 0.5 * np.roll(y, 1)  # has some genuine predictive content
    forecast_unrestricted[0] = 0.0
    result = clark_west(y, forecast_restricted, forecast_unrestricted, h=1)
    assert result.p_value < 0.5  # should lean towards favouring the unrestricted model


def test_giacomini_white_rejects_predictable_relative_performance() -> None:
    rng = np.random.RandomState(3)
    n = 2000
    # Construct a loss differential that is strongly predictable from an
    # instrument (regime indicator), which GW should detect even though the
    # unconditional mean of d_t can be small.
    regime = (rng.rand(n) > 0.5).astype(float)
    d = np.where(regime == 1, 1.0, -1.0) + rng.randn(n) * 0.1
    loss_a = d
    loss_b = np.zeros(n)
    instruments = np.column_stack([np.ones(n), regime])
    result = giacomini_white(loss_a, loss_b, instruments)
    assert result.p_value < 0.01


def test_giacomini_white_constant_instrument_reduces_to_something_like_dm() -> None:
    rng = np.random.RandomState(4)
    n = 1500
    loss_a = np.abs(rng.randn(n)) + 0.5
    loss_b = np.abs(rng.randn(n)) * 0.2
    result = giacomini_white(loss_a, loss_b, np.ones(n))
    assert result.p_value < 0.01
    assert result.n_instruments == 1
