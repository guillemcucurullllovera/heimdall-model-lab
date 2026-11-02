from __future__ import annotations

import numpy as np

from model_lab.eval.multiple_testing import benjamini_hochberg, holm_bonferroni


def test_holm_bonferroni_against_known_textbook_example() -> None:
    # Classic worked example (e.g. Wikipedia's Holm-Bonferroni worked case):
    # p = [0.01, 0.04, 0.03, 0.005], alpha=0.05 -> sorted [0.005, 0.01, 0.03, 0.04]
    # thresholds [0.05/4, 0.05/3, 0.05/2, 0.05/1] = [0.0125, 0.01667, 0.025, 0.05]
    # 0.005<=0.0125 reject; 0.01<=0.01667 reject; 0.03<=0.025 FAIL -> stop (0.04 also not rejected)
    p = [0.01, 0.04, 0.03, 0.005]
    reject = holm_bonferroni(p, alpha=0.05)
    assert reject == [True, False, False, True]


def test_holm_bonferroni_all_pass_when_all_p_tiny() -> None:
    p = [1e-6, 1e-7, 1e-8]
    assert holm_bonferroni(p, alpha=0.05) == [True, True, True]


def test_holm_bonferroni_none_pass_when_all_p_large() -> None:
    p = [0.5, 0.6, 0.9]
    assert holm_bonferroni(p, alpha=0.05) == [False, False, False]


def test_holm_bonferroni_single_test_reduces_to_plain_alpha() -> None:
    assert holm_bonferroni([0.04], alpha=0.05) == [True]
    assert holm_bonferroni([0.06], alpha=0.05) == [False]


def test_holm_bonferroni_empty_input() -> None:
    assert holm_bonferroni([], alpha=0.05) == []


def test_holm_bonferroni_is_at_least_as_strict_as_uncorrected_alpha() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.1, size=24)
    reject = holm_bonferroni(p, alpha=0.05)
    uncorrected = (p < 0.05).tolist()
    # Holm can never reject something uncorrected-alpha=0.05 would not also reject
    for r, u in zip(reject, uncorrected, strict=True):
        assert (not r) or u


def test_benjamini_hochberg_against_known_textbook_example() -> None:
    # p = [0.01, 0.04, 0.03, 0.005], alpha=0.05, m=4
    # sorted [0.005, 0.01, 0.03, 0.04], thresholds [0.0125, 0.025, 0.0375, 0.05]
    # all four sorted p-values <= their threshold -> all rejected (BH is less strict than Holm here)
    p = [0.01, 0.04, 0.03, 0.005]
    reject = benjamini_hochberg(p, alpha=0.05)
    assert reject == [True, True, True, True]


def test_benjamini_hochberg_is_never_stricter_than_holm_bonferroni() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, size=24)
    holm = holm_bonferroni(p, alpha=0.05)
    bh = benjamini_hochberg(p, alpha=0.05)
    for h, b in zip(holm, bh, strict=True):
        assert (not h) or b  # anything Holm rejects, BH (less conservative) also rejects


def test_benjamini_hochberg_empty_input() -> None:
    assert benjamini_hochberg([], alpha=0.05) == []
