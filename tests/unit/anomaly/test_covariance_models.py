from __future__ import annotations

import numpy as np
import pytest

from model_lab.anomaly.covariance_models import mahalanobis_d2_series
from model_lab.anomaly.isolation_forest_model import isolation_forest_score_series
from model_lab.eval.leakage import check_prefix_invariance


def _synthetic_panel(n: int = 400, k: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(k), 0.0001 * np.eye(k), size=n)


def test_mahalanobis_flags_a_planted_joint_shock() -> None:
    r = _synthetic_panel()
    r[300] = [0.2, -0.2, 0.2]  # a large, coherent joint move
    res = mahalanobis_d2_series(r, fold_train_ends=[100, 250], estimator="ledoit_wolf")
    valid = ~np.isnan(res.mahalanobis_d2)
    assert valid[300]
    percentile = (res.mahalanobis_d2[valid] <= res.mahalanobis_d2[300]).mean()
    assert percentile > 0.99


def test_mahalanobis_no_leakage_prefix_invariance() -> None:
    r = _synthetic_panel()
    fold_train_ends = [100, 250]

    def feature_fn(data: np.ndarray) -> np.ndarray:
        local_ends = [e for e in fold_train_ends if e < len(data)]
        return mahalanobis_d2_series(data, local_ends, estimator="ledoit_wolf").mahalanobis_d2

    result = check_prefix_invariance(feature_fn, r, checkpoints=[260, 300, 350], atol=1e-6)
    assert result.all_passed, result


def test_mahalanobis_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="unknown estimator"):
        mahalanobis_d2_series(_synthetic_panel(), [100], estimator="not_a_real_estimator")


def test_mahalanobis_nan_before_first_foldable_train_end() -> None:
    r = _synthetic_panel(n=200, k=4)
    res = mahalanobis_d2_series(r, fold_train_ends=[150], estimator="empirical")
    assert np.all(np.isnan(res.mahalanobis_d2[:150]))
    assert not np.any(np.isnan(res.mahalanobis_d2[150:]))


def test_all_four_covariance_estimators_produce_finite_scores() -> None:
    from model_lab.anomaly.covariance_models import ESTIMATORS

    r = _synthetic_panel(n=300, k=3)
    for name in ESTIMATORS:
        res = mahalanobis_d2_series(r, fold_train_ends=[150], estimator=name)
        valid = res.mahalanobis_d2[~np.isnan(res.mahalanobis_d2)]
        assert len(valid) > 0
        assert np.all(np.isfinite(valid))
        assert np.all(valid >= 0)  # Mahalanobis distance squared is non-negative


def test_isolation_forest_flags_a_planted_joint_shock() -> None:
    r = _synthetic_panel(n=400, k=3, seed=1)
    r[300] = [0.3, -0.3, 0.3]
    res = isolation_forest_score_series(r, fold_train_ends=[100, 250])
    valid = ~np.isnan(res.anomaly_score)
    assert valid[300]
    percentile = (res.anomaly_score[valid] <= res.anomaly_score[300]).mean()
    assert percentile > 0.95
