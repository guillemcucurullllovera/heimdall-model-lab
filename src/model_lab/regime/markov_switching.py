"""2-state Gaussian Markov-switching mean+variance model, periodic-refit
walk-forward (same discipline as `vol.garch_models`): parameters estimated
ONCE per calendar-year fold on strictly-prior (and purged, see below)
data; `statsmodels`' `MarkovRegression.filter(params)` (the Hamilton
filter applied to GIVEN, FIXED params) then produces causal FILTERED
(never smoothed) regime probabilities for the fold's test window, using
the full series' actual observations up to each t but never re-estimating
parameters on them -- the exact same "fix params, then filter/forecast
forward" pattern `vol.garch_models` uses via `arch`'s `.fix()`.

Regime labels from `statsmodels` are NOT guaranteed consistent across
separate fits (which fitted regime is "0" vs "1" is arbitrary) -- this
module canonicalizes per fold by sorting regimes by their fitted variance,
so `low_vol_prob`/`high_vol_prob` are always consistently labelled.

## Remediation (independent adversarial review, findings P1-2, P2-5, P2-7)

Three fixes landed together here because they all touch the same fold
loop:

1. **One-step-ahead transition (P1-2).** The prior version used the
   FILTERED probability `P(regime=high | data through t)` directly as an
   approximate forecast for day t+1, justified by a claim that fitted
   transition persistence was "≥0.99 throughout" every fold. That claim
   was independently checked and found FALSE -- persistence is as low as
   0.00 (a degenerate GLD fold) and commonly 0.4-0.9 elsewhere, nowhere
   near a blanket 0.99 floor. Fixed properly (not merely re-worded): the
   real one-step transition matrix (`MarkovRegression.regime_transition_
   matrix`, column-stochastic: `tm[i,j] = P(state_t=i | state_{t-1}=j)`)
   is now used to propagate the filtered distribution forward exactly one
   step (`predicted = tm @ filtered_t`) before it is used as a forecast --
   the mathematically correct, causal one-step-ahead regime forecast, not
   an approximation that happens to be adequate only when persistence is
   high.

2. **Purge (P2-5).** Parameters are now estimated on
   `log_returns[:train_end_idx - purge_days]`, matching
   `vol.garch_models`' identical fix and rationale -- the forecast origin
   (`train_end_idx`) is unchanged, only what data may be used to ESTIMATE
   params shrinks.

3. **Degenerate-fold guard (P2-7).** A fold whose fitted transition matrix
   shows near-zero persistence for either regime (e.g. the real GLD
   fold-1 case: `p11 ~= 0.0000`, a boundary/non-identified MLE solution
   that nonetheless reports the optimizer's own `converged=True`) is not a
   genuine, economically meaningful 2-regime structure -- it is flagged
   `well_identified=False` and excluded from `high_vol_prob`/
   `regime_variance_forecast` (left NaN, the same treatment as a fit that
   failed to converge at all), rather than silently blended in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

NAN = float("nan")

# A fold's fitted transition matrix must show at least this much self-
# persistence for BOTH regimes to be treated as a genuine, well-identified
# 2-regime structure. Deliberately low and disclosed: real, legitimately
# lower-persistence folds exist in this data (e.g. TLT/IEF's first two
# folds sit around 0.43-0.49) and should NOT be discarded -- this threshold
# targets only clearly pathological/boundary solutions (the real GLD
# fold-1 case sits at 0.0000), not merely "less persistent than average".
MIN_PERSISTENCE = 0.05


@dataclass(frozen=True)
class RegimeFoldFit:
    train_end_idx: int
    fit_end_idx: int
    low_vol_variance: float
    high_vol_variance: float
    low_vol_mean: float
    high_vol_mean: float
    low_vol_persistence: float
    high_vol_persistence: float
    converged: bool
    well_identified: bool  # False if numerically converged but degenerate (see MIN_PERSISTENCE)


@dataclass(frozen=True)
class MarkovSwitchingResult:
    high_vol_prob: np.ndarray  # one-step-ahead P(high-vol regime at t+1 | data through t), NaN where uncovered
    regime_variance_forecast: np.ndarray  # high_vol_prob-weighted blend of the two regimes' own fitted variances
    fold_fits: list[RegimeFoldFit]


def fit_and_filter_2state(
    log_returns: np.ndarray, fold_train_ends: list[int], *, purge_days: int = 1, min_persistence: float = MIN_PERSISTENCE
) -> MarkovSwitchingResult:
    n = len(log_returns)
    high_vol_prob = np.full(n, NAN, dtype=float)
    regime_variance_forecast = np.full(n, NAN, dtype=float)
    fold_fits: list[RegimeFoldFit] = []
    sorted_ends = sorted(set(fold_train_ends))

    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 150 or train_end_idx >= n:
            continue
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        fit_end_idx = max(train_end_idx - purge_days, 150)
        try:
            mod_train = MarkovRegression(
                log_returns[:fit_end_idx], k_regimes=2, trend="c", switching_variance=True
            )
            res_train = mod_train.fit(disp=False)
        except (ValueError, np.linalg.LinAlgError):
            fold_fits.append(
                RegimeFoldFit(
                    train_end_idx, fit_end_idx, float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), False, False,
                )
            )
            continue

        # statsmodels param order for MarkovRegression(k_regimes=2, trend='c',
        # switching_variance=True): [p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]].
        params = res_train.params
        means = np.array([params[2], params[3]])
        variances = np.array([params[4], params[5]])
        low_idx, high_idx = np.argsort(variances)  # canonical: 0=low-vol, 1=high-vol

        # Real one-step transition matrix (column-stochastic:
        # tm[i,j] = P(state_t=i | state_{t-1}=j)) -- confirmed live during
        # remediation via a labelled synthetic check (column sums == 1.0
        # exactly). `p[0->0]`/`p[1->0]` name the SAME quantities this
        # matrix reports; computed via the model's own method rather than
        # re-deriving from params[0:2] by hand, so it is exercised the same
        # way regardless of statsmodels' own regime-index convention.
        tm = mod_train.regime_transition_matrix(params)[:, :, 0]
        persistence = np.diag(tm)  # [P(0->0), P(1->1)] in statsmodels' own regime order
        low_persistence = float(persistence[low_idx])
        high_persistence = float(persistence[high_idx])
        well_identified = low_persistence >= min_persistence and high_persistence >= min_persistence

        fold_fits.append(
            RegimeFoldFit(
                train_end_idx=train_end_idx,
                fit_end_idx=fit_end_idx,
                low_vol_variance=float(variances[low_idx]),
                high_vol_variance=float(variances[high_idx]),
                low_vol_mean=float(means[low_idx]),
                high_vol_mean=float(means[high_idx]),
                low_vol_persistence=low_persistence,
                high_vol_persistence=high_persistence,
                converged=True,
                well_identified=well_identified,
            )
        )
        if not well_identified:
            continue  # leave high_vol_prob/regime_variance_forecast as NaN for this fold's window

        mod_full = MarkovRegression(log_returns, k_regimes=2, trend="c", switching_variance=True)
        res_full = mod_full.filter(params)
        filtered = res_full.filtered_marginal_probabilities  # shape (n, 2), statsmodels' OWN (arbitrary) regime order

        for t in range(train_end_idx, next_boundary):
            # One-step-ahead forecast for t+1, using only information through t:
            # propagate the FILTERED distribution at t forward exactly one step
            # via the real transition matrix, rather than using filtered[t]
            # itself as an (unjustified, persistence-dependent) approximation.
            predicted_next = tm @ filtered[t]
            p_high = float(predicted_next[high_idx])
            high_vol_prob[t] = p_high
            regime_variance_forecast[t] = p_high * variances[high_idx] + (1.0 - p_high) * variances[low_idx]

    return MarkovSwitchingResult(
        high_vol_prob=high_vol_prob, regime_variance_forecast=regime_variance_forecast, fold_fits=fold_fits
    )
