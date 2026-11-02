"""Independent release tests for scientific edge cases, not empirical claims."""
import numpy as np

from model_lab.vol.combination import median_variance_combination


def test_ensemble_handles_missing_constituent_without_changing_finite_median():
    actual = median_variance_combination(np.array([1., np.nan]), np.array([3., np.nan]), np.array([np.nan, np.nan]))
    assert actual[0] == 2.
    assert np.isnan(actual[1])


def test_ensemble_is_prefix_invariant():
    rng = np.random.default_rng(61)
    members = [rng.uniform(.001, .01, 100) for _ in range(3)]
    np.testing.assert_array_equal(median_variance_combination(*members)[:40],
                                  median_variance_combination(*(m[:40] for m in members)))
