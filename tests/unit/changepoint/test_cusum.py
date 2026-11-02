from __future__ import annotations

import numpy as np

from model_lab.changepoint.cusum import cusum_detect
from model_lab.eval.leakage import check_prefix_invariance, future_shuffle_test


def test_cusum_detects_a_planted_mean_shift() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.01, 300), rng.normal(0.05, 0.01, 200)])  # 5-sigma mean shift
    res = cusum_detect(x, window=60, drift=0.5, threshold=5.0)
    alarms_after_shift = np.where(res.alarm[300:320])[0]
    assert len(alarms_after_shift) > 0, "expected CUSUM to alarm shortly after a clear 5-sigma mean shift"


def test_cusum_no_leakage_prefix_invariance() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 300)

    def feature_fn(data: np.ndarray) -> np.ndarray:
        return cusum_detect(data, window=30, drift=0.5, threshold=5.0).g_plus

    result = check_prefix_invariance(feature_fn, x)
    assert result.all_passed, result


def test_cusum_no_leakage_future_shuffle() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 300)
    assert future_shuffle_test(lambda d: cusum_detect(d, window=30).g_plus, x, checkpoint=150)


def test_cusum_quiet_series_rarely_alarms() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 500)
    res = cusum_detect(x, window=60, drift=0.5, threshold=5.0)
    assert res.alarm.sum() <= 3  # a handful of false positives is fine, a flood is not
