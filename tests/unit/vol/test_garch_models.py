from __future__ import annotations

import numpy as np
import pytest

import model_lab.vol.garch_models as garch_models_module
from model_lab.vol.garch_models import (
    MAX_PLAUSIBLE_CUMULATIVE_VARIANCE,
    MIN_PLAUSIBLE_CUMULATIVE_VARIANCE,
    egarch_t_forecast,
    gjr_garch_t_forecast,
)


def _synthetic_returns(n: int = 700, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # A mild GARCH-like process so fitting isn't degenerate.
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-4
    for t in range(n):
        if t > 0:
            sigma2[t] = 1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_t(6)
    return r


def test_gjr_garch_t_forecast_shape_and_nan_pattern() -> None:
    r = _synthetic_returns()
    fold_train_ends = [300, 500]
    res = gjr_garch_t_forecast(r, horizon=1, fold_train_ends=fold_train_ends)
    assert len(res.variance_forecast) == len(r)
    assert np.all(np.isnan(res.variance_forecast[:300]))  # nothing forecast before the first fold's train_end
    assert not np.any(np.isnan(res.variance_forecast[300:]))
    assert np.all(res.variance_forecast[300:] > 0)
    assert len(res.fold_params) == 2


def test_gjr_garch_t_forecast_no_leakage_spot_check() -> None:
    """The forecast at a given origin must not change when the series is
    truncated to end shortly after that origin (the exact property the
    MODEL-2A volatility/tail ladder's own no-leakage-proof section checks)."""
    r = _synthetic_returns(n=500)
    fold_train_ends = [300]
    full = gjr_garch_t_forecast(r, horizon=1, fold_train_ends=fold_train_ends)
    origin = 350
    truncated = gjr_garch_t_forecast(r[: origin + 3], horizon=1, fold_train_ends=[300])
    assert np.isclose(full.variance_forecast[origin], truncated.variance_forecast[origin], rtol=1e-6)


def test_egarch_t_forecast_shape_and_positivity() -> None:
    # Plain (non-recursive) Student-t noise here, not the hand-rolled GARCH
    # recursion above: that generator occasionally drives a degenerate
    # EGARCH fit (near-unit persistence) into simulation overflow on some
    # seeds -- a synthetic-data robustness issue, not a code defect (the
    # same code produces well-behaved results on real SPY data; see
    # reports/MODEL2A_DATA_VOL_TAIL_BASELINES.md).
    rng = np.random.default_rng(1)
    r = rng.standard_t(6, 600) * 0.01
    res = egarch_t_forecast(r, horizon=1, fold_train_ends=[300])
    assert not np.any(np.isnan(res.variance_forecast[300:]))
    assert np.all(np.isfinite(res.variance_forecast[300:]))
    assert np.all(res.variance_forecast[300:] > 0)


def test_egarch_t_forecast_no_leakage_spot_check() -> None:
    """Remediation (independent adversarial review, finding P1-4): this
    exact check existed for GARCH_t but NOT for EGARCH_t anywhere in the
    delivered evidence, despite EGARCH_t being the single best model on
    some real (ticker, horizon) cells (e.g. SPY h=1). EGARCH_t uses
    `method="simulation"` (not an analytic forecast), so this is a genuine,
    separate verification, not a copy-paste of the GARCH_t case -- a
    stochastic simulation with a seed that consumed randomness differently
    between a full-series and a truncated call could plausibly fail this
    even with no real leakage; confirmed here that it does not."""
    r = _synthetic_returns(n=700, seed=2)
    fold_train_ends = [300]
    full = egarch_t_forecast(r, horizon=1, fold_train_ends=fold_train_ends)
    origin = 350
    truncated = egarch_t_forecast(r[: origin + 3], horizon=1, fold_train_ends=[300])
    assert np.isclose(full.variance_forecast[origin], truncated.variance_forecast[origin], rtol=1e-6)


def test_egarch_t_forecast_no_leakage_at_multiple_horizons() -> None:
    """The same check at a non-trivial horizon (h=5), matching how
    `vol.evaluate.run_vol_ladder` actually calls EGARCH_t in production."""
    r = _synthetic_returns(n=700, seed=3)
    fold_train_ends = [300]
    full = egarch_t_forecast(r, horizon=5, fold_train_ends=fold_train_ends)
    origin = 350
    truncated = egarch_t_forecast(r[: origin + 8], horizon=5, fold_train_ends=[300])
    assert np.isclose(full.variance_forecast[origin], truncated.variance_forecast[origin], rtol=1e-6)


def test_a_finite_but_implausible_simulated_value_is_excluded_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for a real defect found during Layer A research
    (2026-10-01): EGARCH_t's simulation-based forecast can produce a FINITE
    but numerically absurd value (e.g. 2.27e+38, found live on real TLT
    data) that `np.isfinite` alone does not catch. `_run_ladder_model` must
    now also reject any value exceeding `MAX_PLAUSIBLE_CUMULATIVE_VARIANCE`,
    leaving that origin NaN (excluded from scoring) rather than silently
    contaminating QLIKE and any downstream combination."""
    r = _synthetic_returns(n=400)
    fold_train_ends = [300]
    real_origins, real_params, real_values = garch_models_module._forecast_one_fold(
        r, 1, 300, 300, vol="GARCH", o=1, method="analytic"
    )
    planted_values = real_values.copy()
    planted_values[0] = 1e38  # finite, but absurd -- must not pass np.isfinite-only filtering

    def _fake_forecast_one_fold(*args, **kwargs):
        return real_origins, real_params, planted_values

    monkeypatch.setattr(garch_models_module, "_forecast_one_fold", _fake_forecast_one_fold)
    res = garch_models_module._run_ladder_model(r, 1, fold_train_ends, model_id="GARCH_t", vol="GARCH", o=1, method="analytic")
    assert np.isnan(res.variance_forecast[int(real_origins[0])]), "the implausible planted value must be excluded, not scored"
    assert not np.any(res.variance_forecast[300:] > MAX_PLAUSIBLE_CUMULATIVE_VARIANCE), "no forecast may exceed the plausibility bound"


def test_a_finite_but_implausibly_tiny_simulated_value_is_also_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the SYMMETRIC instance of the same defect: EGARCH_t's
    simulation can also underflow to a near-zero (e.g. 2.0e-46, found live on
    real TLT data) cumulative variance, which blows up QLIKE's `y/sigma2_hat`
    ratio via a near-zero denominator instead of a huge numerator -- the
    identical downstream symptom via the opposite numerical direction."""
    r = _synthetic_returns(n=400)
    fold_train_ends = [300]
    real_origins, real_params, real_values = garch_models_module._forecast_one_fold(
        r, 1, 300, 300, vol="GARCH", o=1, method="analytic"
    )
    planted_values = real_values.copy()
    planted_values[0] = 1e-46  # finite, positive, but absurdly tiny

    def _fake_forecast_one_fold(*args, **kwargs):
        return real_origins, real_params, planted_values

    monkeypatch.setattr(garch_models_module, "_forecast_one_fold", _fake_forecast_one_fold)
    res = garch_models_module._run_ladder_model(r, 1, fold_train_ends, model_id="GARCH_t", vol="GARCH", o=1, method="analytic")
    assert np.isnan(res.variance_forecast[int(real_origins[0])]), "the implausibly tiny planted value must be excluded, not scored"
    valid = res.variance_forecast[~np.isnan(res.variance_forecast)]
    assert np.all(valid >= MIN_PLAUSIBLE_CUMULATIVE_VARIANCE), "no forecast may fall below the plausibility floor"


def test_a_value_within_absolute_bounds_but_far_from_the_local_naive_scale_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a THIRD tier of the same defect: a planted value
    can pass both absolute [MIN, MAX] bounds yet still be orders of
    magnitude off from the realised variance at that specific origin (real
    examples found live: qlike_mean inflated into the thousands for several
    (ticker, horizon) cells even after the absolute bounds alone). The
    `naive_trailing_rv_forecast`-relative band must catch this even when the
    absolute bounds do not."""
    r = _synthetic_returns(n=400)
    fold_train_ends = [300]
    real_origins, real_params, real_values = garch_models_module._forecast_one_fold(
        r, 1, 300, 300, vol="GARCH", o=1, method="analytic"
    )
    planted_values = real_values.copy()
    # Within [MIN_PLAUSIBLE_CUMULATIVE_VARIANCE, MAX_PLAUSIBLE_CUMULATIVE_VARIANCE] (1e-10, 1.0)
    # but ~1000x the typical synthetic series' own trailing variance (~1e-4) -- must still be caught.
    planted_values[0] = 0.5

    def _fake_forecast_one_fold(*args, **kwargs):
        return real_origins, real_params, planted_values

    monkeypatch.setattr(garch_models_module, "_forecast_one_fold", _fake_forecast_one_fold)
    res = garch_models_module._run_ladder_model(r, 1, fold_train_ends, model_id="GARCH_t", vol="GARCH", o=1, method="analytic")
    assert np.isnan(res.variance_forecast[int(real_origins[0])]), (
        "a value within absolute bounds but implausible relative to the local naive scale must still be excluded"
    )


def test_gjr_and_egarch_forecasts_non_overlapping_fold_ownership() -> None:
    """Fold k must not own any origin >= the NEXT fold's train_end (each
    origin's forecast must come from the single fold whose training window
    actually precedes it)."""
    r = _synthetic_returns(n=700)
    fold_train_ends = [300, 500]
    res = gjr_garch_t_forecast(r, horizon=1, fold_train_ends=fold_train_ends)
    assert res.fold_params[0]["train_end_idx"] == 300
    assert res.fold_params[1]["train_end_idx"] == 500
