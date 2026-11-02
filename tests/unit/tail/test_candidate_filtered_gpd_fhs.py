from __future__ import annotations

import numpy as np

from model_lab.tail.candidate_filtered_gpd_fhs import combo_filtered_gpd_fhs_forecast
from model_lab.tail.conditional import gjr_t_filtered_gpd_fhs_forecast


def _synthetic_returns(n: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-4
    for t in range(n):
        if t > 0:
            sigma2[t] = 1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_t(6)
    return r


def test_combo_filtered_gpd_fhs_produces_positive_var_es() -> None:
    r = _synthetic_returns()
    fold_train_ends = [300, 500]
    res = combo_filtered_gpd_fhs_forecast(r, fold_train_ends)
    for p in (0.05, 0.025, 0.01):
        valid = ~np.isnan(res.var_forecast[p])
        assert valid.sum() > 0
        assert np.all(res.var_forecast[p][valid] > 0)
        assert np.all(res.es_forecast[p][valid] > 0)
        assert np.all(res.es_forecast[p][valid] >= res.var_forecast[p][valid]), "ES must be at least as large as VaR"


def test_combo_and_incumbent_differ_but_share_the_same_tail_shape_fit() -> None:
    """The ablation is isolated to the vol SCALE, not the tail shape -- the
    two models' fold_fits (the GPD fit on GARCH_t's own standardized
    residuals) must be IDENTICAL; only the scaled VaR/ES values may differ."""
    r = _synthetic_returns(seed=3)
    fold_train_ends = [300, 500]
    incumbent = gjr_t_filtered_gpd_fhs_forecast(r, fold_train_ends)
    candidate = combo_filtered_gpd_fhs_forecast(r, fold_train_ends)
    assert incumbent.fold_fits == candidate.fold_fits, "tail shape must be identical -- only vol scaling differs"
    # The scaled outputs need not be identical (different vol inputs) -- confirm they are NOT
    # trivially identical (i.e. the ablation actually changes something).
    p = 0.05
    valid = ~np.isnan(incumbent.var_forecast[p]) & ~np.isnan(candidate.var_forecast[p])
    assert not np.allclose(incumbent.var_forecast[p][valid], candidate.var_forecast[p][valid])


def test_combo_filtered_gpd_fhs_no_leakage_spot_check() -> None:
    """Prefix-truncation check, same discipline as every other model in
    this project: a forecast at a given origin must be unchanged when the
    series is truncated right after it."""
    r = _synthetic_returns(n=800, seed=7)
    fold_train_ends = [300, 500]
    origin = 650
    full = combo_filtered_gpd_fhs_forecast(r, fold_train_ends)
    truncated = combo_filtered_gpd_fhs_forecast(r[: origin + 5], [e for e in fold_train_ends if e <= origin])
    for p in (0.05, 0.025, 0.01):
        assert np.isclose(full.var_forecast[p][origin], truncated.var_forecast[p][origin], rtol=1e-6)
        assert np.isclose(full.es_forecast[p][origin], truncated.es_forecast[p][origin], rtol=1e-6)
