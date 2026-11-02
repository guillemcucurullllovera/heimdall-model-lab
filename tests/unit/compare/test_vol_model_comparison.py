from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from model_lab.compare.vol_model_comparison import compare_vol_models
from model_lab.vol.data import ReturnSeries
from model_lab.vol.evaluate import COMBO_MODEL_ID


def _synthetic_series(n: int = 900, seed: int = 0) -> ReturnSeries:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-4
    for t in range(n):
        if t > 0:
            sigma2[t] = 1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_t(6)

    dates = []
    d = date(2018, 1, 2)
    while len(dates) < n + 1:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return ReturnSeries(
        ticker="SYN", dataset_id="syn_raw", dates=dates, closes=closes, highs=closes, lows=closes,
        opens=closes, log_returns=r, data_fingerprint="s" * 64,
    )


def test_compare_vol_models_structure_and_bounds() -> None:
    series = _synthetic_series()
    result = compare_vol_models(series, horizon=1, min_train_groups=2, spa_reps=200, mcs_reps=200)
    assert result.ticker == "SYN"
    assert result.horizon == 1
    assert set(result.model_ids) == {"naive_trailing_rv", "EWMA_0.94", "GARCH_t", "EGARCH_t", COMBO_MODEL_ID}
    assert 0.0 <= result.spa_p_value_consistent <= 1.0
    assert result.mcs_superior_set  # MCS should never return an empty set
    assert set(result.mcs_superior_set).issubset(set(result.model_ids))
    assert result.n_oos > 0
