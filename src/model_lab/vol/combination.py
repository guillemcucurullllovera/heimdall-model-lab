"""Forecast combination (Bates & Granger 1969, "The Combination of Forecasts",
Operational Research Quarterly 20(4):451-468; simple equal-weight averaging,
their eq. for the case of no estimated/optimized weights -- the "combination
puzzle" literature, e.g. Timmermann 2006, "Forecast Combinations" in the
Handbook of Economic Forecasting, repeatedly finds equal-weighting hard to
beat out-of-sample despite its naivety, because weight-ESTIMATION error in
more sophisticated schemes often outweighs any theoretical efficiency gain).

This is a research probe, not a new fitted model: it takes forecasts that
are ALREADY causal (each produced by a walk-forward model evaluated
elsewhere in this repository) and combines them with a fixed, data-
independent rule (equal weights). It cannot introduce leakage that wasn't
already present in its inputs, because it performs no estimation of its own
-- there is nothing to fit, and therefore nothing that could see the future.
"""

from __future__ import annotations

import warnings

import numpy as np


def equal_weight_variance_combination(*forecasts: np.ndarray) -> np.ndarray:
    """Elementwise arithmetic mean of two or more variance-forecast arrays
    (Bates & Granger 1969's equal-weight case). NaN at position i in ANY
    input forecast propagates to NaN in the combination at position i (a
    combination is only defined where every member has an opinion).

    Fail-closed by design/contract: unlike `median_variance_combination`
    below, this function makes no claim of NaN-robustness and none is
    implemented. Currently unused by any evidence-producing script in this
    repository (`median_variance_combination` is used instead throughout
    `vol/evaluate.py` and `compare/vol_model_comparison.py`); retained for
    its own test coverage and as the literal Bates & Granger (1969)
    equal-weight baseline the median variant is compared against in the
    docstring below."""
    if len(forecasts) < 2:
        raise ValueError("combination requires at least 2 forecasts")
    stacked = np.stack(forecasts, axis=0)
    return np.mean(stacked, axis=0)


def median_variance_combination(*forecasts: np.ndarray) -> np.ndarray:
    """Elementwise median of two or more variance-forecast arrays -- the
    standard robustness variant of equal-weight combination discussed
    alongside the mean in the same literature (e.g. Stock & Watson 2004,
    "Combination Forecasts of Output Growth in a Seven-Country Data Set",
    Journal of Forecasting 23(6):405-430, compare mean/median/trimmed-mean
    combination and find the median materially more robust whenever any
    single member model is occasionally badly behaved).

    Used in this repository in preference to the mean (`vol/evaluate.py`,
    `compare/vol_model_comparison.py`) after a real, live example of exactly
    that failure mode: EGARCH_t's simulation-based forecast was found to
    occasionally produce a finite-but-numerically-absurd value (see
    `vol/garch_models.py`'s own `MAX_PLAUSIBLE_CUMULATIVE_VARIANCE` guard,
    added for the same discovery), which `vol/garch_models.py` now DETECTS
    and EXCLUDES (replaces with NaN) at the origin where it occurs, rather
    than letting it dominate an equal-weight mean of 3 members.

    EXPLICIT NaN CONTRACT (remediation of independent-review finding P1-C:
    the prior implementation used `np.median`, under which a single NaN
    member propagates to NaN in the output -- the opposite of robustness,
    and never actually exercised the "ignore one wild member" claim above
    because the wild EGARCH_t value never reaches this function at all: it
    is already converted to NaN upstream before combination runs). The
    INTENDED and now IMPLEMENTED contract is NaN-ROBUST, not fail-closed:
    this function uses `np.nanmedian`, so the combination at position i is
    the median of whichever members are NOT NaN at i, and is NaN only if
    EVERY member is NaN at i. This makes the combination genuinely robust
    to the actual failure mode this repository produces (a member filtered
    to NaN by its own upstream plausibility check), not merely to a
    same-shape-but-finite outlier that this codebase's data flow never lets
    through. The classic "median ignores a wild-but-finite outlier among
    3 members" property (Stock & Watson's own framing) still holds exactly
    as before whenever all members ARE finite -- this change only expands
    the combination's domain to also be defined when up to (n-1) of n
    members are NaN, never narrows it."""
    if len(forecasts) < 2:
        raise ValueError("combination requires at least 2 forecasts")
    stacked = np.stack(forecasts, axis=0)
    with warnings.catch_warnings():
        # "All-NaN slice encountered" is the EXPECTED, correct way numpy
        # reports position i where every member is NaN; the result (NaN) is
        # exactly the contract above, not a condition to surface as a
        # warning on every call.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(stacked, axis=0)
