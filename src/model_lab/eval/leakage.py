"""FeatureSet leakage harness — MODEL1_foundation_contracts.md §9 rule 7 /
AT-05: "a leakage harness (prefix-invariance, future-shuffle, seeded leaks)
runs on every FeatureSet."

Prefix-invariance: a causally-computed feature at time t must depend ONLY
on data[0:t+1] — computing the feature on the full series and reading index
t must equal computing it on the truncated series data[:t+1] and reading
its last value. A feature function that fails this check is using future
information (a "leak").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PrefixInvarianceResult:
    checkpoints: list[int]
    passed_at: list[bool]
    all_passed: bool
    max_abs_diff: float


def check_prefix_invariance(feature_fn: FeatureFn, data: np.ndarray, checkpoints: list[int] | None = None, *, atol: float = 1e-9) -> PrefixInvarianceResult:
    """For each checkpoint t, compare feature_fn(data)[t] against
    feature_fn(data[:t+1])[-1]. A mismatch means the feature at t depended
    on data beyond index t -- a leak."""
    n = len(data)
    if checkpoints is None:
        checkpoints = sorted(set(np.linspace(1, n - 1, min(10, n - 1)).astype(int).tolist()))

    full_features = feature_fn(data)
    passed_at = []
    max_diff = 0.0
    for t in checkpoints:
        truncated = feature_fn(data[: t + 1])
        full_val = full_features[t]
        trunc_val = truncated[-1]
        if np.isnan(full_val) and np.isnan(trunc_val):
            ok = True
            diff = 0.0
        else:
            diff = abs(float(full_val) - float(trunc_val))
            ok = diff <= atol
        max_diff = max(max_diff, diff)
        passed_at.append(ok)

    return PrefixInvarianceResult(checkpoints=checkpoints, passed_at=passed_at, all_passed=all(passed_at), max_abs_diff=max_diff)


def future_shuffle_test(feature_fn: FeatureFn, data: np.ndarray, *, seed: int = 0, checkpoint: int | None = None) -> bool:
    """Shuffle only the data STRICTLY AFTER `checkpoint` and confirm the
    feature value AT `checkpoint` is unchanged -- a second, independent way
    to detect a leak (complementary to prefix-invariance): if the feature
    at t depends on anything after t, shuffling the tail changes it."""
    n = len(data)
    if checkpoint is None:
        checkpoint = n // 2
    rng = np.random.RandomState(seed)
    shuffled = data.copy()
    tail = shuffled[checkpoint + 1 :].copy()
    rng.shuffle(tail)
    shuffled[checkpoint + 1 :] = tail

    original_val = feature_fn(data)[checkpoint]
    shuffled_val = feature_fn(shuffled)[checkpoint]
    if np.isnan(original_val) and np.isnan(shuffled_val):
        return True
    return abs(float(original_val) - float(shuffled_val)) <= 1e-9


@dataclass(frozen=True)
class SeededLeakSuiteResult:
    n_variants: int
    n_caught: int
    caught: list[bool]
    labels: list[str]


def seeded_leak_suite(feature_fn_factory: Callable[[], list[tuple[str, FeatureFn]]], data: np.ndarray) -> SeededLeakSuiteResult:
    """`feature_fn_factory()` returns a list of (label, feature_fn) pairs,
    each a DELIBERATELY-LEAKY variant of a base causal feature. Returns how
    many of them the harness (prefix-invariance) correctly flags as leaking."""
    variants = feature_fn_factory()
    caught = []
    labels = []
    for label, fn in variants:
        result = check_prefix_invariance(fn, data)
        caught.append(not result.all_passed)  # "caught" means the harness correctly detected the leak
        labels.append(label)
    return SeededLeakSuiteResult(n_variants=len(variants), n_caught=sum(caught), caught=caught, labels=labels)
