from __future__ import annotations

import numpy as np
import pytest

from model_lab.eval.scores import (
    brier_score,
    crps_gaussian,
    crps_gaussian_mixture,
    crps_student_t,
    fz0_joint_var_es,
    log_score_gaussian,
    pinball_loss,
    qlike,
)


def test_qlike_minimised_at_truth() -> None:
    rng = np.random.RandomState(0)
    true_var = np.abs(rng.randn(500)) + 1.0
    values = [qlike(true_var, true_var * mult).value for mult in [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 2.0]]
    assert values.index(min(values)) == 3  # mult=1.0 (the truth) minimises QLIKE


def test_qlike_proxy_floor_avoids_infinite_loss() -> None:
    out = qlike(np.array([0.0]), np.array([1.0]))
    assert np.isfinite(out.value)


def test_log_score_gaussian_minimised_at_true_sigma() -> None:
    rng = np.random.RandomState(1)
    y = rng.randn(2000) * 2.0
    mu = np.zeros_like(y)
    sigmas = [1.0, 1.5, 1.9, 2.0, 2.1, 2.5, 3.0]
    values = [log_score_gaussian(y, mu, np.full_like(y, s)).value for s in sigmas]
    assert sigmas[values.index(min(values))] == pytest.approx(2.0, abs=0.2)


def test_crps_gaussian_minimised_at_true_sigma() -> None:
    rng = np.random.RandomState(2)
    y = rng.randn(3000) * 1.5
    mu = np.zeros_like(y)
    sigmas = [0.8, 1.2, 1.4, 1.5, 1.6, 1.8, 2.2]
    values = [crps_gaussian(y, mu, np.full_like(y, s)).value for s in sigmas]
    assert sigmas[values.index(min(values))] == pytest.approx(1.5, abs=0.2)


def test_crps_student_t_reduces_to_reasonable_values() -> None:
    rng = np.random.RandomState(3)
    y = stats_t_sample(rng, 3000)
    out = crps_student_t(y, np.zeros_like(y), np.ones_like(y), nu=5.0)
    assert out.value > 0
    assert np.isfinite(out.value)


def stats_t_sample(rng: np.random.RandomState, n: int) -> np.ndarray:
    from scipy import stats

    return stats.t.rvs(df=5, size=n, random_state=rng)


def test_crps_gaussian_mixture_matches_single_gaussian_when_degenerate() -> None:
    rng = np.random.RandomState(4)
    y = rng.randn(1000)
    single = crps_gaussian(y, np.zeros_like(y), np.ones_like(y)).value
    mixture = crps_gaussian_mixture(y, [0.5, 0.5], [np.zeros_like(y), np.zeros_like(y)], [np.ones_like(y), np.ones_like(y)]).value
    assert mixture == pytest.approx(single, rel=1e-9)


def test_brier_score_propriety() -> None:
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    true_p = y.astype(float)
    other_p = np.full_like(true_p, 0.5)
    assert brier_score(y, true_p).value < brier_score(y, other_p).value


def test_pinball_loss_symmetry_at_median() -> None:
    rng = np.random.RandomState(5)
    y = rng.randn(1000)
    out_median_true = pinball_loss(y, np.zeros_like(y), tau=0.5).value
    out_median_off = pinball_loss(y, np.full_like(y, 0.5), tau=0.5).value
    assert out_median_true < out_median_off


def test_fz0_joint_var_es_minimised_near_truth() -> None:
    rng = np.random.RandomState(6)
    y = rng.standard_t(df=5, size=5000) * 1.0
    alpha = 0.05
    true_var = float(np.quantile(y, alpha))
    true_es = float(y[y <= true_var].mean())
    candidates = [(true_var * m, true_es * m) for m in [0.7, 0.9, 1.0, 1.1, 1.3]]
    values = [fz0_joint_var_es(y, np.full_like(y, v), np.full_like(y, e), alpha).value for v, e in candidates]
    assert values.index(min(values)) == 2  # m=1.0 (the truth)


def test_fz0_requires_negative_es() -> None:
    with pytest.raises(ValueError):
        fz0_joint_var_es(np.array([0.0]), np.array([-1.0]), np.array([1.0]), 0.05)
