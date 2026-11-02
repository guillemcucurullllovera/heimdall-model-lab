from __future__ import annotations

import numpy as np
import pytest

from model_lab.vol.combination import equal_weight_variance_combination, median_variance_combination


def test_equal_weight_combination_is_the_arithmetic_mean() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    c = np.array([2.0, 3.0, 4.0])
    out = equal_weight_variance_combination(a, b, c)
    np.testing.assert_allclose(out, np.array([2.0, 3.0, 4.0]))


def test_nan_in_any_member_propagates_to_the_combination() -> None:
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    out = equal_weight_variance_combination(a, b)
    assert np.isnan(out[1])
    assert not np.isnan(out[0])
    assert not np.isnan(out[2])


def test_requires_at_least_two_forecasts() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        equal_weight_variance_combination(np.array([1.0, 2.0]))


def test_combination_of_identical_forecasts_is_that_forecast() -> None:
    a = np.array([5.0, 6.0, 7.0])
    out = equal_weight_variance_combination(a, a, a)
    np.testing.assert_allclose(out, a)


# -- median_variance_combination: the function actually used in production
# (`vol/evaluate.py`, `compare/vol_model_comparison.py`). Remediation of
# independent-review finding P1-C: these were previously entirely absent --
# only the unused `equal_weight_variance_combination` had direct tests.


def test_median_combination_of_three_finite_values_ignores_a_wild_outlier() -> None:
    """The literature-cited property (Stock & Watson 2004): with 3 finite
    members, the median is exactly the middle value, so a single member
    many orders of magnitude off the other two has NO effect at all --
    unlike the mean, which it would dominate."""
    normal_a = np.array([1.2e-4])
    normal_b = np.array([1.4e-4])
    wild_outlier = np.array([2.27e38])  # the real magnitude reported live for EGARCH_t's defect
    out = median_variance_combination(normal_a, normal_b, wild_outlier)
    np.testing.assert_allclose(out, np.array([1.4e-4]))  # the middle of the 3, not pulled at all
    mean_out = (normal_a + normal_b + wild_outlier) / 3
    assert mean_out[0] > 1e37  # the mean, by contrast, is completely dominated by the outlier


def test_median_combination_ignores_a_single_nan_member() -> None:
    """The failure mode this repository actually produces: EGARCH_t's own
    plausibility filter (`vol/garch_models.py`) converts an implausible
    value to NaN BEFORE it ever reaches this function -- so the member that
    needs to be tolerated here is NaN, not a wild-but-finite outlier (see
    the test above for that, separate, property). With exactly one NaN
    member among 3, the combination must equal the median of the other
    two (= their mean, since 2 values have no unique middle)."""
    ewma = np.array([1.2e-4, 3.0e-4])
    garch_t = np.array([1.4e-4, 5.0e-4])
    egarch_t_filtered = np.array([np.nan, 7.0e-4])
    out = median_variance_combination(ewma, garch_t, egarch_t_filtered)
    np.testing.assert_allclose(out, np.array([(1.2e-4 + 1.4e-4) / 2, 5.0e-4]))


def test_median_combination_is_nan_only_when_every_member_is_nan() -> None:
    a = np.array([np.nan, 1.0, np.nan])
    b = np.array([np.nan, 2.0, 3.0])
    c = np.array([np.nan, 3.0, 5.0])
    out = median_variance_combination(a, b, c)
    assert np.isnan(out[0])  # all 3 NaN -> NaN
    np.testing.assert_allclose(out[1:], np.array([2.0, 4.0]))  # 0/2 NaN, 1/3 NaN resp. -> real median


def test_median_combination_requires_at_least_two_forecasts() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        median_variance_combination(np.array([1.0, 2.0]))


def test_median_combination_of_identical_forecasts_is_that_forecast() -> None:
    a = np.array([5.0, 6.0, 7.0])
    out = median_variance_combination(a, a, a)
    np.testing.assert_allclose(out, a)


def test_median_combination_emits_no_warning_on_an_all_nan_column() -> None:
    """The expected, correct NaN-for-an-all-NaN-position case must not
    surface numpy's "All-NaN slice encountered" RuntimeWarning on every
    call -- that warning is suppressed deliberately (see implementation),
    not accidentally silenced project-wide."""
    import warnings

    a = np.array([np.nan])
    b = np.array([np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning now raises
        out = median_variance_combination(a, b)
    assert np.isnan(out[0])
