from __future__ import annotations

import numpy as np

from model_lab.eval.mcs_spa import hansen_spa, model_confidence_set, romano_wolf_stepm


def test_spa_finds_a_clearly_superior_model() -> None:
    rng = np.random.RandomState(0)
    n = 1000
    bench = np.abs(rng.randn(n)) + 1.0
    better = np.abs(rng.randn(n)) * 0.2
    worse = np.abs(rng.randn(n)) + 1.5
    models = np.column_stack([better, worse])
    result = hansen_spa(bench, models, reps=300, seed=1)
    assert not result.benchmark_not_rejected
    assert result.n == n


def test_spa_does_not_reject_when_benchmark_is_best() -> None:
    rng = np.random.RandomState(2)
    n = 1000
    bench = np.abs(rng.randn(n)) * 0.1
    worse1 = np.abs(rng.randn(n)) + 1.0
    worse2 = np.abs(rng.randn(n)) + 1.5
    models = np.column_stack([worse1, worse2])
    result = hansen_spa(bench, models, reps=300, seed=3)
    assert result.benchmark_not_rejected


def test_stepm_superior_set_contains_the_known_best_model() -> None:
    rng = np.random.RandomState(4)
    n = 1000
    bench = np.abs(rng.randn(n)) + 1.0
    better = np.abs(rng.randn(n)) * 0.2
    worse = np.abs(rng.randn(n)) + 1.5
    models = np.column_stack([better, worse])
    result = romano_wolf_stepm(bench, models, alpha=0.10, reps=300, seed=5)
    assert 0 in result.superior_model_indices


def test_mcs_excludes_a_clearly_worse_model() -> None:
    rng = np.random.RandomState(6)
    n = 1000
    good1 = np.abs(rng.randn(n)) * 0.2
    good2 = np.abs(rng.randn(n)) * 0.22
    bad = np.abs(rng.randn(n)) + 2.0
    losses = np.column_stack([good1, good2, bad])
    result = model_confidence_set(losses, alpha=0.10, reps=300, seed=7)
    assert 2 in result.excluded_model_indices
