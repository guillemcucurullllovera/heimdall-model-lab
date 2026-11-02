from __future__ import annotations

import numpy as np

from model_lab.eval.var_es_backtests import (
    acerbi_szekely_es_statistic,
    christoffersen_lr_cc,
    christoffersen_lr_ind,
    hit_sequence,
    kupiec_lr_uc,
)


def test_lr_cc_equals_lr_uc_plus_lr_ind() -> None:
    rng = np.random.RandomState(0)
    hits = (rng.rand(2000) < 0.05).astype(int)
    p = 0.05
    uc = kupiec_lr_uc(hits, p)
    ind = christoffersen_lr_ind(hits)
    cc = christoffersen_lr_cc(hits, p)
    assert abs(cc.lr_stat - (uc.lr_stat + ind.lr_stat)) < 1e-9
    assert cc.df == 2


def test_kupiec_correctly_calibrated_series_has_high_p_value_on_average() -> None:
    rng = np.random.RandomState(1)
    n_trials = 200
    pvals = []
    for _ in range(n_trials):
        hits = (rng.rand(1000) < 0.05).astype(int)
        pvals.append(kupiec_lr_uc(hits, 0.05).p_value)
    # size close to nominal: roughly 5% of trials should have p<0.05 under H0
    rejection_rate = np.mean(np.array(pvals) < 0.05)
    assert 0.0 <= rejection_rate <= 0.12


def test_kupiec_rejects_badly_miscalibrated_series() -> None:
    rng = np.random.RandomState(2)
    hits = (rng.rand(1000) < 0.15).astype(int)  # actual rate 15% vs nominal 5%
    result = kupiec_lr_uc(hits, 0.05)
    assert result.p_value < 0.001


def test_lr_ind_rejects_clustered_violations() -> None:
    # Construct a strongly clustered hit sequence (violations come in runs).
    hits = np.zeros(1000, dtype=int)
    for start in range(0, 1000, 100):
        hits[start : start + 10] = 1  # 10 consecutive violations in a block
    result = christoffersen_lr_ind(hits)
    assert result.p_value < 0.01


def test_lr_ind_does_not_reject_iid_hits() -> None:
    rng = np.random.RandomState(3)
    hits = (rng.rand(3000) < 0.05).astype(int)
    result = christoffersen_lr_ind(hits)
    assert result.p_value > 0.01


def test_hit_sequence() -> None:
    losses = np.array([1.0, 2.0, 0.5, 3.0])
    var = np.array([1.5, 1.5, 1.5, 1.5])
    hits = hit_sequence(losses, var)
    assert list(hits) == [0, 1, 0, 1]


def test_acerbi_szekely_near_zero_for_well_specified_es() -> None:
    rng = np.random.RandomState(4)
    n = 20000
    losses = rng.standard_normal(n)
    p = 0.05
    var_true = float(np.quantile(losses, 1 - p))
    tail = losses[losses > var_true]
    es_true = float(tail.mean())
    var_forecast = np.full(n, var_true)
    es_forecast = np.full(n, es_true)
    result = acerbi_szekely_es_statistic(losses, var_forecast, es_forecast, p)
    assert abs(result.z_stat) < 0.5


def test_acerbi_szekely_flags_underestimated_es() -> None:
    rng = np.random.RandomState(5)
    n = 20000
    losses = rng.standard_normal(n)
    p = 0.05
    var_true = float(np.quantile(losses, 1 - p))
    var_forecast = np.full(n, var_true)
    es_forecast = np.full(n, var_true * 1.05)  # ES underestimated (too close to VaR)
    result = acerbi_szekely_es_statistic(losses, var_forecast, es_forecast, p)
    assert result.z_stat > 0.1
