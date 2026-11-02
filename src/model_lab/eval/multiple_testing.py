"""Multiple-testing corrections for a FAMILY of independent hypothesis
tests -- distinct from, and complementary to, `eval/mcs_spa.py`'s
Hansen SPA / Romano-Wolf StepM / Model Confidence Set, which correct for
comparing MULTIPLE MODELS against one benchmark on the SAME data sample.

This module is for the other, equally real multiple-comparison problem in
this repository: a single DM (or any other) p-value computed independently
once per (ticker, horizon) CELL, where each cell is a different data sample.
Remediation of independent-review finding P2-2: the MODEL-2G report applied
an informal Bonferroni-style check to its regime-conditional-value probe
(6 tests) but reported its two headline 24-cell tallies -- "best single
model beats naive at DM p<0.05" and "combination beats GARCH_t at DM
p<0.05" -- with no equivalent correction, despite being exactly the same
kind of family (many independent per-cell tests, naively read at the
uncorrected alpha=0.05 threshold).
"""

from __future__ import annotations

import numpy as np


def holm_bonferroni(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[bool]:
    """Holm (1979) step-down procedure. Controls the familywise error rate
    (FWER) at `alpha` -- the probability of AT LEAST ONE false positive
    across the whole family -- uniformly more powerful than a flat
    Bonferroni correction (`alpha / m`) while giving the same FWER
    guarantee. Returns, in the ORIGINAL input order, whether each
    hypothesis is rejected (True) under the correction.

    Procedure: sort p-values ascending as p_(1) <= ... <= p_(m); reject
    p_(1)..p_(k) where k is the largest index with p_(i) <= alpha / (m - i + 1)
    for all i <= k (equivalently: walk up from the smallest p-value,
    stop at the first failure)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        threshold = alpha / (m - i)
        if sorted_p[i] <= threshold:
            reject_sorted[i] = True
        else:
            break  # step-down: once one fails, every subsequent (larger) p-value also fails
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg (1995) step-up procedure. Controls the FALSE
    DISCOVERY RATE (FDR, the expected proportion of false positives AMONG
    the rejections) at `alpha` -- a less conservative criterion than
    Holm-Bonferroni's FWER, appropriate when some false positives are
    tolerable as long as most rejections are real. Returns, in the
    ORIGINAL input order, whether each hypothesis is rejected (True).

    Procedure: sort ascending as p_(1) <= ... <= p_(m); find the largest k
    such that p_(k) <= (k / m) * alpha; reject all p_(1)..p_(k)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passing = np.where(sorted_p <= thresholds)[0]
    k = int(passing.max()) + 1 if len(passing) > 0 else 0
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[:k] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()
