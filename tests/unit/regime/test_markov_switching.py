from __future__ import annotations

import numpy as np

from model_lab.regime.markov_switching import fit_and_filter_2state


def _two_regime_series(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0, 0.01, 400), rng.normal(0, 0.04, 300), rng.normal(0, 0.01, 300)]
    )


def test_high_vol_prob_rises_during_the_planted_high_variance_segment() -> None:
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[300, 600])
    assert np.nanmean(res.high_vol_prob[420:480]) > 0.8


def test_high_vol_prob_falls_after_reverting_to_the_low_variance_regime() -> None:
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[300, 600])
    assert np.nanmean(res.high_vol_prob[750:850]) < 0.2


def test_fold_fits_report_low_variance_less_than_high_variance() -> None:
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[300, 600])
    for f in res.fold_fits:
        if f.converged:
            assert f.low_vol_variance < f.high_vol_variance


def test_nan_before_first_foldable_train_end() -> None:
    # train_end=600 (not 400): training must include at least some of the
    # high-variance segment (which starts at 400) for the 2-regime fit to be
    # genuinely well-identified -- training on 400 (all-calm) data alone
    # correctly trips the new degenerate-fold guard (see
    # test_degenerate_fold_is_flagged_and_excluded below), which is the
    # guard working as intended, not a bug in this unrelated test.
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[600])
    assert np.all(np.isnan(res.high_vol_prob[:600]))
    assert not np.any(np.isnan(res.high_vol_prob[600:]))


def test_degenerate_fold_is_flagged_and_excluded() -> None:
    """Remediation (independent adversarial review, finding P2-7): a fold
    trained on data containing no genuine second regime yet (here: the
    first 400 points, all from the calm segment, before the high-variance
    segment starting at 400 has occurred) produces a numerically
    `converged=True` but economically degenerate fit (near-zero
    persistence for the spurious "second regime") -- this must be flagged
    `well_identified=False` and excluded from `high_vol_prob`/
    `regime_variance_forecast` (left NaN), not silently blended in."""
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[400])
    fit = res.fold_fits[0]
    assert fit.converged is True
    assert fit.well_identified is False
    assert min(fit.low_vol_persistence, fit.high_vol_persistence) < 0.05
    assert np.all(np.isnan(res.high_vol_prob[400:]))
    assert np.all(np.isnan(res.regime_variance_forecast[400:]))


def test_regime_variance_forecast_is_a_convex_blend_of_the_two_regime_variances() -> None:
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[600])
    fit = res.fold_fits[0]
    valid = ~np.isnan(res.regime_variance_forecast)
    assert np.all(res.regime_variance_forecast[valid] >= fit.low_vol_variance - 1e-12)
    assert np.all(res.regime_variance_forecast[valid] <= fit.high_vol_variance + 1e-12)


def test_fit_end_idx_is_purged_by_purge_days() -> None:
    """Remediation (independent adversarial review, finding P2-5): the
    parameter fit must use a window strictly shorter than the nominal
    train_end_idx by `purge_days`."""
    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[600], purge_days=7)
    fit = res.fold_fits[0]
    assert fit.fit_end_idx == fit.train_end_idx - 7


def test_one_step_ahead_prob_actually_differs_from_the_same_day_filtered_prob() -> None:
    """Remediation (independent adversarial review, finding P1-2): the
    forecast stored at t must be the ONE-STEP-AHEAD transitioned
    probability, not the same-day filtered probability reused verbatim.
    This test independently recomputes the raw filtered probability via
    `statsmodels` directly (bypassing `fit_and_filter_2state` entirely) and
    asserts it is NOT what `high_vol_prob` reports -- discriminating this
    fix from the pre-fix behaviour, not just checking output is finite."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    x = _two_regime_series()
    res = fit_and_filter_2state(x, fold_train_ends=[600])
    fit = res.fold_fits[0]
    assert fit.well_identified
    assert fit.low_vol_persistence < 0.999 or fit.high_vol_persistence < 0.999

    mod_train = MarkovRegression(x[: fit.fit_end_idx], k_regimes=2, trend="c", switching_variance=True)
    res_train = mod_train.fit(disp=False)
    params = res_train.params
    variances = np.array([params[4], params[5]])
    high_idx = int(np.argmax(variances))
    mod_full = MarkovRegression(x, k_regimes=2, trend="c", switching_variance=True)
    filtered = mod_full.filter(params).filtered_marginal_probabilities

    t = 650
    same_day_filtered_high_prob = float(filtered[t, high_idx])
    reported = res.high_vol_prob[t]
    assert not np.isclose(reported, same_day_filtered_high_prob, atol=1e-9), (
        f"high_vol_prob[{t}]={reported} must differ from the raw same-day filtered "
        f"probability {same_day_filtered_high_prob} -- if they match exactly, the one-step "
        "transition is not actually being applied"
    )
