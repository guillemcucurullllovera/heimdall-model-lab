from __future__ import annotations

import numpy as np

from model_lab.gate_v2.formulas import pbo_cscv


def test_pbo_combinations_count_uses_math_comb_not_the_papers_typo() -> None:
    # The PBO paper's own worked example states S=16 -> "12,780" combinations;
    # the correct value is C(16,8) = 12,870 (a documented typo in the paper).
    rng = np.random.RandomState(0)
    perf = rng.randn(64, 3) * 0.01
    result = pbo_cscv(perf, S=16)
    assert result.n_combinations == 12870


def test_pbo_high_for_pure_noise_trials() -> None:
    # With trials that are pure noise (no real signal), the in-sample best
    # trial should frequently fail to generalise out-of-sample -> high PBO.
    rng = np.random.RandomState(1)
    perf = rng.randn(200, 10) * 0.01
    result = pbo_cscv(perf, S=8)
    assert result.pbo > 0.3


def test_pbo_low_for_one_dominant_trial() -> None:
    # One trial has a strong, persistent edge; it should generalise well
    # in-sample -> out-of-sample, giving a low PBO.
    rng = np.random.RandomState(2)
    perf = rng.randn(200, 10) * 0.005
    perf[:, 0] += 0.01  # trial 0 has a strong, persistent edge
    result = pbo_cscv(perf, S=8)
    assert result.pbo < 0.3


def test_pbo_result_bounds() -> None:
    rng = np.random.RandomState(3)
    perf = rng.randn(100, 5) * 0.01
    result = pbo_cscv(perf, S=10)
    assert 0.0 <= result.pbo <= 1.0
    assert len(result.logits) == result.n_combinations
