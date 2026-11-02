"""AT-08: LR_uc/LR_ind/LR_cc size and power simulations; FZ0 propriety."""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.eval.scores import fz0_joint_var_es
from model_lab.eval.var_es_backtests import christoffersen_lr_cc, kupiec_lr_uc

pytestmark = pytest.mark.acceptance


def test_lr_uc_size_near_nominal() -> None:
    rng = np.random.RandomState(0)
    n_trials = 300
    p = 0.05
    rejections = 0
    for _ in range(n_trials):
        hits = (rng.rand(1000) < p).astype(int)
        if kupiec_lr_uc(hits, p).p_value < 0.05:
            rejections += 1
    size = rejections / n_trials
    assert 0.0 <= size <= 0.10  # near-nominal, loose bound for a fast MC budget


def test_lr_cc_power_against_miscalibrated_rate() -> None:
    rng = np.random.RandomState(1)
    n_trials = 200
    p_nominal = 0.05
    rejections = 0
    for _ in range(n_trials):
        hits = (rng.rand(1000) < 0.12).astype(int)  # actual rate more than double nominal
        if christoffersen_lr_cc(hits, p_nominal).p_value < 0.05:
            rejections += 1
    power = rejections / n_trials
    assert power > 0.8  # high power against a large miscalibration


def test_lr_cc_power_against_clustering() -> None:
    rng = np.random.RandomState(2)
    n_trials = 200
    p = 0.05
    rejections = 0
    for _ in range(n_trials):
        hits = np.zeros(1000, dtype=int)
        # same overall rate as nominal, but clustered into a few large runs
        n_violations = int(1000 * p)
        starts = rng.choice(1000 - 20, size=max(1, n_violations // 10), replace=False)
        for s in starts:
            hits[s : s + 10] = 1
        if christoffersen_lr_cc(hits, p).p_value < 0.05:
            rejections += 1
    power = rejections / n_trials
    assert power > 0.5  # clustering should be detectable most of the time


def test_fz0_propriety_minimised_near_truth() -> None:
    rng = np.random.RandomState(3)
    y = rng.standard_t(df=6, size=8000)
    alpha = 0.05
    true_var = float(np.quantile(y, alpha))
    true_es = float(y[y <= true_var].mean())
    grid = np.linspace(0.7, 1.3, 13)
    values = [fz0_joint_var_es(y, np.full_like(y, true_var * m), np.full_like(y, true_es * m), alpha).value for m in grid]
    best_m = grid[int(np.argmin(values))]
    assert best_m == pytest.approx(1.0, abs=0.15)
