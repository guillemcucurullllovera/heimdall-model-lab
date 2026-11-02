"""AT-05: prefix-invariance holds for all FeatureSets; seeded-leak suite
catches >= 9 of 10."""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.eval.leakage import check_prefix_invariance, future_shuffle_test, seeded_leak_suite

pytestmark = pytest.mark.acceptance


def _causal_trailing_mean(window: int) -> callable:
    def fn(data: np.ndarray) -> np.ndarray:
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - window + 1)
            out[t] = data[lo : t + 1].mean()
        return out

    return fn


def test_clean_causal_feature_passes_prefix_invariance() -> None:
    rng = np.random.RandomState(0)
    data = rng.randn(200)
    fn = _causal_trailing_mean(10)
    result = check_prefix_invariance(fn, data)
    assert result.all_passed, result


def test_clean_causal_feature_passes_future_shuffle_test() -> None:
    rng = np.random.RandomState(1)
    data = rng.randn(200)
    fn = _causal_trailing_mean(10)
    assert future_shuffle_test(fn, data, checkpoint=50)


def _leaky_variants() -> list[tuple[str, callable]]:
    base_window = 10

    def leak_centered_window(data: np.ndarray) -> np.ndarray:
        # Uses a CENTERED window (includes future points).
        out = np.full(len(data), np.nan)
        half = base_window // 2
        for t in range(len(data)):
            lo, hi = max(0, t - half), min(len(data), t + half + 1)
            out[t] = data[lo:hi].mean()
        return out

    def leak_global_normalisation(data: np.ndarray) -> np.ndarray:
        # Normalises by the FULL series' std (look-ahead).
        std = data.std()
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - base_window + 1)
            out[t] = data[lo : t + 1].mean() / (std + 1e-9)
        return out

    def leak_shifted_forward(data: np.ndarray) -> np.ndarray:
        # Off-by-one: uses data[t+1] instead of data[t] as the window end.
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            hi = min(len(data), t + 2)
            lo = max(0, hi - base_window)
            out[t] = data[lo:hi].mean()
        return out

    def leak_full_series_cumsum_reversed(data: np.ndarray) -> np.ndarray:
        # Computed from the reverse-cumulative sum of the WHOLE series.
        rev_cumsum = np.cumsum(data[::-1])[::-1]
        n = len(data)
        out = np.full(n, np.nan)
        for t in range(n):
            out[t] = rev_cumsum[t] / (n - t)
        return out

    def leak_global_detrend(data: np.ndarray) -> np.ndarray:
        # Detrends against the full-series linear trend, then takes a trailing mean.
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        detrended = data - trend
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - base_window + 1)
            out[t] = detrended[lo : t + 1].mean()
        return out

    def leak_peek_next_value(data: np.ndarray) -> np.ndarray:
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - base_window + 1)
            peek = data[t + 1] if t + 1 < len(data) else data[t]
            out[t] = 0.9 * data[lo : t + 1].mean() + 0.1 * peek
        return out

    def leak_full_series_rank(data: np.ndarray) -> np.ndarray:
        # Rank-transforms against the FULL series' distribution.
        order = data.argsort().argsort()
        ranks = order / (len(data) - 1)
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - base_window + 1)
            out[t] = ranks[lo : t + 1].mean()
        return out

    def leak_global_max_scale(data: np.ndarray) -> np.ndarray:
        scale = np.abs(data).max()
        out = np.full(len(data), np.nan)
        for t in range(len(data)):
            lo = max(0, t - base_window + 1)
            out[t] = data[lo : t + 1].mean() / (scale + 1e-9)
        return out

    def leak_backward_ewma(data: np.ndarray) -> np.ndarray:
        # Exponential smoothing applied backward across the whole series,
        # then read forward (each point depends on everything after it).
        alpha = 0.3
        n = len(data)
        smoothed = np.zeros(n)
        acc = data[-1]
        for i in range(n - 1, -1, -1):
            acc = alpha * data[i] + (1 - alpha) * acc
            smoothed[i] = acc
        return smoothed

    def leak_full_window_then_index(data: np.ndarray) -> np.ndarray:
        # Computes a centered rolling window over the WHOLE series via
        # convolution (look-ahead by construction), then indexes it.
        kernel = np.ones(base_window) / base_window
        padded = np.pad(data, (base_window // 2, base_window // 2), mode="edge")
        conv = np.convolve(padded, kernel, mode="valid")
        return conv[: len(data)]

    return [
        ("centered_window", leak_centered_window),
        ("global_normalisation", leak_global_normalisation),
        ("shifted_forward_off_by_one", leak_shifted_forward),
        ("full_series_reverse_cumsum", leak_full_series_cumsum_reversed),
        ("global_detrend", leak_global_detrend),
        ("peek_next_value", leak_peek_next_value),
        ("full_series_rank_transform", leak_full_series_rank),
        ("global_max_scale", leak_global_max_scale),
        ("backward_ewma", leak_backward_ewma),
        ("full_window_convolution", leak_full_window_then_index),
    ]


def test_seeded_leak_suite_catches_at_least_9_of_10() -> None:
    rng = np.random.RandomState(2)
    data = rng.randn(300) + np.linspace(0, 2, 300)  # trend included, to make detrend-leak meaningfully different
    result = seeded_leak_suite(_leaky_variants, data)
    assert result.n_variants == 10
    assert result.n_caught >= 9, (
        f"only caught {result.n_caught}/10: "
        f"{[label for label, caught in zip(result.labels, result.caught, strict=True) if not caught]}"
    )
