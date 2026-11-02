from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from model_lab.tail.gpd import fit_gpd_tail, gpd_var_es


def test_fit_gpd_tail_recovers_known_shape_on_synthetic_exceedances() -> None:
    """Losses drawn i.i.d. from an EXACT GPD(true_xi, true_beta): by the
    GPD's own threshold-stability property, exceedances above ANY higher
    threshold are themselves GPD-distributed with the SAME shape parameter
    -- so fitting `fit_gpd_tail` at a 10% tail quantile on this series
    should recover `true_xi` closely, with no artificial discontinuity at
    the threshold biasing the MLE."""
    rng = np.random.default_rng(0)
    true_xi, true_beta = 0.2, 0.01
    losses = genpareto.rvs(true_xi, scale=true_beta, size=2000, random_state=rng)
    returns = -losses

    fit = fit_gpd_tail(returns, tail_quantile=0.10)
    assert fit.n == 2000
    assert fit.n_exceedances == pytest.approx(200, abs=20)
    assert abs(fit.xi - true_xi) < 0.15


def test_gpd_var_es_ordering_and_monotonicity() -> None:
    """ES must exceed VaR (it's the expectation beyond the VaR quantile),
    and both must grow as the tail probability shrinks (deeper into the
    tail -> larger loss)."""
    rng = np.random.default_rng(1)
    losses_gen = genpareto.rvs(0.1, scale=0.01, size=600, random_state=rng) + 0.02
    returns = np.concatenate([-losses_gen, rng.uniform(-0.01, 0.01, size=400)])
    fit = fit_gpd_tail(returns, tail_quantile=0.10)

    var_05, es_05 = gpd_var_es(fit, 0.05)
    var_01, es_01 = gpd_var_es(fit, 0.01)
    assert es_05 > var_05
    assert es_01 > var_01
    assert var_01 > var_05  # deeper tail -> larger VaR
    assert es_01 > es_05


def test_gpd_var_es_rejects_nonfinite_mean_tail() -> None:
    from model_lab.tail.gpd import GpdFit

    fit = GpdFit(u=0.02, xi=1.5, beta=0.01, n=1000, n_exceedances=50)
    with pytest.raises(ValueError, match="xi"):
        gpd_var_es(fit, 0.05)


def test_fit_gpd_tail_raises_on_too_few_exceedances() -> None:
    returns = np.random.default_rng(2).normal(0, 0.01, size=50)
    with pytest.raises(ValueError, match="exceedances"):
        fit_gpd_tail(returns, tail_quantile=0.01)  # ~0-1 exceedances out of 50
