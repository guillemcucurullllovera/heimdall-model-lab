"""Pinned regression vectors for the e-process module — from the Gate v2
research extraction (Ramdas-Wang framework)."""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.eval.eprocess import betting_martingale, e_bh, harm_process, ville_crossing_time


def test_betting_martingale_pinned_vector() -> None:
    d = np.array([0.1, -0.05, 0.2, 0, 0.15, -0.1, 0.3, 0.05])
    result = betting_martingale(d)
    assert result.final_e == pytest.approx(1.29107, rel=1e-4)
    assert result.log_final_e == pytest.approx(0.255474, rel=1e-4)
    assert result.n == 8


def test_e_bh_pinned_vector_alpha_05_none_rejected() -> None:
    e = np.array([80, 30, 25, 4, 1])
    result = e_bh(e, alpha=0.05)
    assert result.k_star == 0
    assert result.rejected_indices == []


def test_e_bh_pinned_vector_alpha_10_three_rejected() -> None:
    e = np.array([80, 30, 25, 4, 1])
    result = e_bh(e, alpha=0.10)
    assert result.k_star == 3
    assert sorted(result.rejected_indices) == [0, 1, 2]  # indices of 80, 30, 25


def test_betting_martingale_rejects_out_of_bound_d() -> None:
    with pytest.raises(ValueError):
        betting_martingale(np.array([0.1, 1.5, -0.2]))


def test_betting_martingale_is_nonnegative_supermartingale_under_null() -> None:
    # Under H0 (E[d_t|F_{t-1}] <= 0), E[E_t] should not exceed 1 "on
    # average" across many independent zero-mean-noise paths -- a coarse
    # Monte Carlo sanity check of the supermartingale property, not a tight
    # bound.
    rng = np.random.RandomState(0)
    n_paths = 2000
    finals = []
    for _ in range(n_paths):
        d = np.clip(rng.randn(50) * 0.1, -1, 1)  # mean-zero noise
        finals.append(betting_martingale(d).final_e)
    assert np.mean(finals) < 1.5  # loose bound; true E[E_T] <= 1 under exact H0


def test_ville_crossing_time_detects_a_clear_edge() -> None:
    # A strongly positive, persistent edge should cross a modest threshold.
    d = np.full(200, 0.3)
    result = betting_martingale(d)
    t = ville_crossing_time(result.e_values, threshold=5.0)
    assert t is not None
    assert t <= 200


def test_ville_crossing_time_none_when_never_crossed() -> None:
    d = np.zeros(50)
    result = betting_martingale(d)
    t = ville_crossing_time(result.e_values, threshold=1000.0)
    assert t is None


def test_harm_process_flags_degraded_and_quarantined_thresholds() -> None:
    # A persistently harmful sequence (d_t consistently below -delta) should
    # eventually cross both the DEGRADED (5) and QUARANTINED (20) harm
    # e-value thresholds.
    delta = 0.05
    d = np.full(500, -0.5)  # much worse than the -delta floor
    result = harm_process(d, delta=delta)
    assert result.degraded
    assert result.quarantined


def test_harm_process_stays_low_for_a_benign_sequence() -> None:
    rng = np.random.RandomState(1)
    delta = 0.1
    d = np.clip(rng.randn(500) * 0.05, -1, 1)  # centred near 0, well above -delta
    result = harm_process(d, delta=delta)
    assert not result.quarantined
