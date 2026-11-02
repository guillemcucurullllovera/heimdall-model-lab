"""Regression test for a real bug caught during MODEL-2C development: the
changepoint ("r_t=0") branch originally reused each hypothesis's own
FITTED predictive density for x_t -- the same one used for the "growth"
branch -- instead of the FRESH PRIOR's predictive density. This breaks the
asymmetry that makes BOCPD work at all: it is provable algebraically that
reusing the fitted predictive for both branches collapses
`P(r_t=0 | x_1:t)` to EXACTLY the hazard rate for every t, regardless of
data (the same normalizing sum appears in both the numerator and the total,
and cancels). Confirmed live before the fix: `changepoint_prob` was a flat
constant across an obvious planted variance shift. Fixed in
`changepoint/bocpd.py` by using the prior's own predictive for the
changepoint branch.
"""

from __future__ import annotations

import numpy as np

from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma


def test_changepoint_prob_is_not_a_flat_constant_equal_to_hazard() -> None:
    """The exact discriminator for the bug: under the bug, EVERY entry of
    `changepoint_prob` equals `hazard` bit-for-bit. Real data (or even pure
    noise) must show variation."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 500)
    hazard = 1.0 / 250.0
    res = bocpd_normal_inverse_gamma(x, hazard=hazard)
    assert not np.allclose(res.changepoint_prob, hazard, atol=1e-12), (
        "changepoint_prob is a flat constant equal to hazard -- the exact signature of the "
        "prior-vs-fitted-predictive bug this test guards against"
    )


def test_bocpd_detects_a_planted_variance_shift() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 0.01, 400), rng.normal(0, 0.05, 300)])
    res = bocpd_normal_inverse_gamma(x, hazard=1.0 / 250.0)

    baseline = np.median(res.changepoint_prob[50:390])  # typical level away from any shift
    near_shift = res.changepoint_prob[400:410].max()
    assert near_shift > 10 * baseline, (
        f"expected a clear spike near the planted shift (index 400): baseline={baseline:.5f}, "
        f"near_shift_max={near_shift:.5f}"
    )
    # MAP run length should reset to a small value shortly after the shift.
    assert res.map_run_length[405] < 10


def test_bocpd_map_run_length_grows_linearly_absent_any_shift() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 60)
    res = bocpd_normal_inverse_gamma(x, hazard=1.0 / 250.0)
    # With a low hazard and no real shift, the MAP run length should track
    # "time since start" closely (index t -> run length t, since index 0 is r=1
    # after the first observation in this implementation's convention).
    np.testing.assert_array_equal(res.map_run_length, np.arange(1, 61))


def test_bocpd_output_length_matches_input() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.02, 250)
    res = bocpd_normal_inverse_gamma(x)
    assert len(res.changepoint_prob) == 250
    assert len(res.map_run_length) == 250
    assert np.all((res.changepoint_prob >= 0) & (res.changepoint_prob <= 1))


# --- Remediation (independent adversarial review, finding P1-3): the default
# `beta0` used to be `np.var(x)` over the ENTIRE input array, computed once
# before the online loop -- a real, quantified look-ahead violation of this
# module's own "purely online/causal by construction" claim (on a series with
# a late shock, changepoint_prob at early SHARED indices differed by up to 4x
# between a full-series run and a prefix-only run). Fixed with a causal,
# Welford-updated expanding variance. These tests are the exact discriminator:
# they fail under the old (non-causal) behaviour and pass under the fix.


def test_bocpd_is_exactly_prefix_invariant_with_a_late_shock() -> None:
    """The core causality property: truncating the series after some point
    must not change ANY earlier result, even when a large shock exists later
    in the (unobserved-at-that-time) full series. Under the pre-fix
    `np.var(x)`-over-everything default, this assertion fails with an
    observed max diff of ~0.025 against a ~0.0005-0.004 baseline scale."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.01, 400), rng.normal(0, 0.05, 300), rng.normal(0, 0.15, 50)])
    full = bocpd_normal_inverse_gamma(x, hazard=1.0 / 250.0)
    prefix = bocpd_normal_inverse_gamma(x[:400], hazard=1.0 / 250.0)

    diffs = np.abs(full.changepoint_prob[:400] - prefix.changepoint_prob[:400])
    assert diffs.max() < 1e-12, (
        f"BOCPD is not prefix-invariant: max abs diff over the shared prefix = {diffs.max():.6g} "
        "(expected exactly 0 -- the whole point of an online/causal algorithm)"
    )
    np.testing.assert_array_equal(full.map_run_length[:400], prefix.map_run_length[:400])


def test_bocpd_causal_beta0_uses_only_prior_data_not_full_series() -> None:
    """A second, independent check on the same property via `check_prefix_invariance`
    (the same harness `eval/leakage.py` uses elsewhere in this codebase), at
    several checkpoints, not just one truncation point."""
    from model_lab.eval.leakage import check_prefix_invariance

    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0, 0.01, 300), rng.normal(0, 0.2, 100)])

    def feature_fn(data: np.ndarray) -> np.ndarray:
        return bocpd_normal_inverse_gamma(data, hazard=1.0 / 250.0).changepoint_prob

    result = check_prefix_invariance(feature_fn, x, checkpoints=[50, 150, 250, 299], atol=1e-12)
    assert result.all_passed, result


def test_bocpd_fixed_beta0_is_also_causal_and_data_independent() -> None:
    """The alternative mode (`beta0=<a float>`) must be usable and must not
    depend on `x` at all -- changing `x` after the truncation point must not
    even indirectly perturb it via a shared default."""
    rng = np.random.default_rng(5)
    x1 = rng.normal(0, 0.01, 200)
    x2 = x1.copy()
    x2[150:] = rng.normal(0, 5.0, 50)  # wildly different tail, should not affect the shared prefix
    r1 = bocpd_normal_inverse_gamma(x1, hazard=1.0 / 250.0, beta0=1e-4)
    r2 = bocpd_normal_inverse_gamma(x2, hazard=1.0 / 250.0, beta0=1e-4)
    np.testing.assert_array_equal(r1.changepoint_prob[:150], r2.changepoint_prob[:150])


def test_bocpd_still_detects_a_planted_shift_with_the_causal_default() -> None:
    """The fix must not have broken detection power -- re-run of the
    pre-existing detection test with the new causal beta0 default."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 0.01, 400), rng.normal(0, 0.05, 300)])
    res = bocpd_normal_inverse_gamma(x, hazard=1.0 / 250.0)
    baseline = np.median(res.changepoint_prob[50:390])
    near_shift = res.changepoint_prob[400:410].max()
    assert near_shift > 10 * baseline
