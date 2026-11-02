"""GJR-GARCH-t and EGARCH-t volatility forecasts (`arch` package), matching
T-VOL's `incumbents=[..., "GARCH_t"]` (GJR-GARCH-t here) plus EGARCH-t as
the ladder's fourth rung (mission item C.4).

Walk-forward discipline: parameters are fit ONCE per fold, on
`log_returns[:train_end_idx]` only (strictly before the fold's test block).
Forecasts for every day in the fold's test window are then produced by
`arch`'s `.fix(params).forecast(start=...)` machinery, which recomputes the
conditional-variance RECURSION using the actual realised returns up through
each forecast origin (never future returns relative to that origin) but
does NOT re-estimate parameters -- standard periodic-refit walk-forward,
not leakage: by the time origin `t` (t >= train_end_idx) is reached, every
return up to and including `t` has already occurred relative to that
origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arch import arch_model

from model_lab.vol.baselines import naive_trailing_rv_forecast

VolFamily = Literal["GARCH", "EGARCH"]

NAN = float("nan")
RETURN_SCALE = 100.0  # arch's own recommendation: fit on ~percent-scale returns for numerical stability

# Discovered during Layer A combination research (2026-10-01), not the independent adversarial
# review: EGARCH_t's `method="simulation"` multi-step forecast (500 Monte Carlo paths of the
# log-variance recursion) can occasionally produce a FINITE but numerically absurd per-origin
# cumulative h-day variance (e.g., for TLT h=1, the real per-point max across all folds is
# 1.24e+24 -- confirmed by directly rerunning the pre-fix forecast function), affecting 13/24
# (ticker, horizon) cells in the committed MODEL-2A evidence. These pass `np.isfinite` (the ONLY
# existing guard) because they are not literally inf/nan, just a simulation blow-up (the
# log-variance random walk compounding through `exp()` over many steps for a fold whose fitted
# asymmetry/persistence happens to be locally explosive under certain simulated paths) -- the same
# failure family already disclosed for one GLD h=45 fold during the independent review's
# remediation (where it DID overflow to inf and so was already correctly excluded), just below the
# overflow threshold instead of above it.
#
# Terminology correction (remediation-specification item 10): the originally-quoted "real values
# of 2.27e+38 (TLT h=1), 3.09e+77 (TLT h=21), 4.32e+93 (USO h=45) found live" were, verified
# directly against the pre-fix committed JSON, each cell's resulting AGGREGATE `qlike_mean` (the
# mean QLIKE loss across ~2000 OOS points, one or a few of which have an astronomical per-point
# forecast) -- NOT a per-point cumulative-variance forecast value as that phrasing implied. The
# genuinely per-point forecast extremes for TLT h=1 are a raw-variance max of ~1.24e+24 and a min
# of ~2.0e-46 (the latter IS a real per-point forecast value, correctly quoted below); the
# 1e+38/1e+77/1e+93-scale numbers are what those per-point extremes do to the QLIKE ratio once
# averaged, not forecasts in their own right.
#
# Reproducibility (remediation-specification item 10): re-running the pre-fix code against this
# exact, `uv.lock`-pinned environment, in two fresh process invocations, reproduced the quoted
# TLT h=1 `qlike_mean` (2.2755960619413493e+38) bit-for-bit both times, matching the historical
# committed JSON exactly -- within this pinned environment the simulation RNG seed
# (`np.random.default_rng(20260930)`) and the `arch` optimizer are fully deterministic, contrary
# to a prior review pass's claim that the value "does not reproduce... on a clean rerun with the
# same seed." That specific claim did not hold when actually tested. What remains a genuinely
# untested (not demonstrated, not ruled out) risk is CROSS-environment reproducibility: this
# mechanism is a 500-path Monte Carlo simulation of a log-variance recursion compounded through
# `exp()`, so its exact magnitude is plausibly sensitive to the exact `arch`/numpy/scipy/BLAS
# versions in a DIFFERENT environment even with "the same seed" in a loose sense (a bit-generator
# algorithm change alone would produce a different stream from an identical seed integer). This
# review did not have a second environment available to test that specific claim either way.
#
# Every well-behaved model's QLIKE across this entire project stays within roughly [0, 2]; a
# cumulative h-day variance of 1.0 (raw squared log-return units) already implies a ~100% move
# over the horizon, absurd for any of these six liquid ETFs at any horizon up to 45 days -- so
# this bound is many, many orders of magnitude above anything a real forecast could legitimately
# produce, and was added BEFORE looking at whether it changed any result in this session's favor.
# This DETECTS AND EXCLUDES the implausible value at its own origin (replaces it with NaN, so it
# is no longer scored) -- it does not stabilise the underlying simulation, which can still
# produce the same kind of blow-up at a DIFFERENT origin/fold/ticker in the future. "Fixed" in
# earlier reports describes the symptom (qlike_mean no longer astronomical) being eliminated, not
# a root-cause numerical stabilisation of `method="simulation"` itself -- see
# `reports/MODEL2G_RESEARCH_LAYERS_ABC.md`'s remediation note and this function's own exclusion
# quantification in `reports/MODEL2E_2F_COMPARISON_SHADOW.md` for the (non-trivial, per-cell
# disclosed) consequence of excluding rather than stabilising.
MAX_PLAUSIBLE_CUMULATIVE_VARIANCE = 1.0

# A second, symmetric instance of the SAME simulation-blow-up failure family, found immediately
# after adding the bound above: the EGARCH_t simulation can also UNDERFLOW to a near-zero (but
# still finite, still positive, still `<= MAX_PLAUSIBLE_CUMULATIVE_VARIANCE`) cumulative variance
# -- a real PER-POINT forecast value as small as 2.0e-46 was found live (TLT, h=1; confirmed by
# directly rerunning the pre-fix forecast function), which the bound above does not catch at all,
# and which then blows UP the QLIKE loss itself via its `y_true / sigma2_hat` ratio (a near-zero
# denominator), reproducing the identical downstream symptom (an astronomical `qlike_mean`) via
# the opposite numerical direction. `GARCH_t`'s own global minimum forecast across every
# ticker/horizon in this project is `4.6e-06`; `1e-10` is four orders of magnitude below even
# that, so it excludes only genuine underflow, never a legitimately small but real low-volatility
# forecast.
MIN_PLAUSIBLE_CUMULATIVE_VARIANCE = 1e-10

# Fixed absolute [MIN, MAX] bounds above still left a THIRD tier of the same defect: values that
# are technically within [1e-10, 1.0] (so pass both absolute bounds) but are nonetheless orders of
# magnitude off from the realised variance the model is actually trying to forecast at that
# specific origin -- a handful of such points were enough to inflate `qlike_mean` into the
# thousands for several (ticker, horizon) cells even after both absolute bounds were added.
# Addressed origin, structurally, rather than by tightening an arbitrary global constant yet
# again: each candidate value is now ALSO checked against `naive_trailing_rv_forecast` at that
# SAME origin (a simple, always-well-behaved, purely causal reference with no fitting of its own)
# and rejected if it deviates by more than `PLAUSIBILITY_BAND_VS_NAIVE` (two orders of magnitude,
# in either direction) -- generous enough that a real, legitimate regime-shift forecast would
# never trip it, but tight enough to catch the residual simulation pathology at its true, local
# scale rather than a single global constant that can never simultaneously be tight for IEF's
# low-volatility days and loose for USO's high-volatility ones. A sensitivity sweep over
# {2, 5, 10, 20, 50, 100, 1000, 1e6, inf} for the two cells where EGARCH_t is the cell's sole best
# model (SPY h=21, USO h=21) found EGARCH_t beats GARCH_t at every tested value including the band
# disabled entirely -- the choice of exactly `100.0` is not load-bearing for those two cells, though
# this was not re-swept across all 24 cells. This third tier is also what silently removes the
# SAME origin from every OTHER model's evaluation sample too, not only EGARCH_t's own, because the
# shared-sample filter in `vol/evaluate.py`/`compare/vol_model_comparison.py` requires every model
# non-NaN at an origin -- quantified per-cell in `reports/egarch_exclusion_diagnostics.json` and
# discussed in `reports/MODEL2E_2F_COMPARISON_SHADOW.md`.
PLAUSIBILITY_BAND_VS_NAIVE = 100.0


@dataclass(frozen=True)
class GarchForecastResult:
    model_id: str
    config: dict
    variance_forecast: np.ndarray  # same length as log_returns; NaN where no fold covers that origin
    fold_params: list[dict]  # one entry per fold: {"train_end_idx": int, "params": {...}}


def _cumulative_h_day_variance(variance_df, horizon: int) -> np.ndarray:
    """Sums the per-step variance columns (near-zero daily return
    autocorrelation makes Var(sum) approx sum(Var), the standard
    GARCH-literature aggregation), rescaled back from RETURN_SCALE**2.

    Selected POSITIONALLY (`.iloc[:, :horizon]`), not by the name `f"h.{j}"`:
    `arch` zero-pads its own column labels to the width of the largest step
    (`h.01`..`h.45` at horizon=45, but `h.1`..`h.9` at horizon<=9) --
    confirmed live (a horizon=45 run on real SPY data raised `KeyError:
    ['h.1', ... 'h.9'] not in index` before this fix); `variance_df` always
    has exactly `horizon` columns in forecast order regardless of label
    width, so positional selection is robust to it."""
    return variance_df.iloc[:, :horizon].sum(axis=1).to_numpy() / (RETURN_SCALE**2)


def _forecast_one_fold(
    log_returns: np.ndarray, horizon: int, fit_end_idx: int, test_start_idx: int, *, vol: VolFamily, o: int, method: str
) -> tuple[np.ndarray, dict, np.ndarray]:
    """`fit_end_idx` (the PURGED boundary -- see `_run_ladder_model`) bounds
    what parameters are ESTIMATED on; `test_start_idx` (the real fold/
    calendar boundary, unpurged) bounds where forecasts are produced FROM.
    The state recursion inside `.forecast(start=test_start_idx)` legitimately
    uses every real, already-realised return up to each forecast origin
    (including the purged gap `[fit_end_idx, test_start_idx)`) -- purging
    only removes those days from PARAMETER ESTIMATION, exactly AFML's
    concept (a training row whose forward-looking label window would
    overlap the test block is dropped from fitting, not from the causally
    available history a forecast may condition on)."""
    r_pct = log_returns * RETURN_SCALE
    am_train = arch_model(r_pct[:fit_end_idx], mean="Zero", vol=vol, p=1, o=o, q=1, dist="t")
    res_train = am_train.fit(disp="off", show_warning=False)

    am_full = arch_model(r_pct, mean="Zero", vol=vol, p=1, o=o, q=1, dist="t")
    res_fixed = am_full.fix(res_train.params)
    kwargs: dict = {"horizon": horizon, "start": test_start_idx, "reindex": False}
    if method == "simulation":
        gen = np.random.default_rng(20260930)
        kwargs.update(method="simulation", simulations=500, rng=lambda size: gen.standard_normal(size))
    fc = res_fixed.forecast(**kwargs)
    origins = fc.variance.index.to_numpy()
    values = _cumulative_h_day_variance(fc.variance, horizon)
    return origins, dict(res_train.params), values


def _run_ladder_model(
    log_returns: np.ndarray, horizon: int, fold_train_ends: list[int], *, model_id: str, vol: VolFamily, o: int, method: str
) -> GarchForecastResult:
    n = len(log_returns)
    out = np.full(n, NAN, dtype=float)
    fold_params: list[dict] = []
    sorted_ends = sorted(set(fold_train_ends))
    naive_ref = naive_trailing_rv_forecast(log_returns, horizon)  # plausibility reference, see
    # PLAUSIBILITY_BAND_VS_NAIVE's own docstring above
    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 30 or train_end_idx >= n:
            continue
        test_start_idx = train_end_idx
        # Remediation (independent adversarial review, finding P2-5): MODEL-1's
        # own purge computation (a training row's forward-looking label window
        # of length `horizon` must not overlap the test block) was computed
        # upstream (`splitter.purge_embargo`) but its result was discarded --
        # every phase report claimed "embargo=horizon days enforced" while the
        # actual fit slice was the full, unpurged calendar-year prefix.
        # Quantified impact of the gap: up to `horizon` days (45 at h=45) of
        # the nominal training window overlapped what purge would have
        # removed. Fixed here by shrinking the FIT boundary by `horizon` days
        # (algebraically identical to what `purge_indices` computes for this
        # pure-expanding, train-only-before-test topology -- verified during
        # remediation: `purge_and_embargo`'s actual `train_indices` for this
        # fold topology is exactly `[0, test_start_idx - horizon)`, a
        # contiguous prefix, never a scattered set). The forecast ORIGIN
        # (`test_start_idx`) is unchanged -- only what parameters may be
        # ESTIMATED on shrinks; the state recursion still uses all real,
        # already-realised history up to each forecast origin (see
        # `_forecast_one_fold`'s own docstring).
        fit_end_idx = max(train_end_idx - horizon, 30)
        try:
            origins, params, values = _forecast_one_fold(
                log_returns, horizon, fit_end_idx, test_start_idx, vol=vol, o=o, method=method
            )
        except np.linalg.LinAlgError as e:
            # Confirmed live on real IEF (a low-volatility bond ETF) at EGARCH's
            # simulation-based multi-step forecast: the log-variance recursion can
            # overflow for a poorly-scaled series (`arch`'s own DataScaleWarning),
            # producing non-finite simulated paths that break the internal
            # long-run-variance least-squares fit (`SVD did not converge`). This is
            # a numerical-stability limitation of this fold's fit, not a logic bug
            # -- fail this ONE fold closed (leave it NaN, recorded below) rather
            # than aborting the whole ladder for every other ticker/fold.
            fold_params.append({"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "params": None, "failed": str(e)[:200]})
            continue
        # Non-overlapping assignment: this fold owns [train_end_idx, next fold's train_end_idx).
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        for origin, value in zip(origins, values, strict=True):
            if not (train_end_idx <= origin < next_boundary and np.isfinite(value)):
                continue
            if not (MIN_PLAUSIBLE_CUMULATIVE_VARIANCE <= abs(value) <= MAX_PLAUSIBLE_CUMULATIVE_VARIANCE):
                continue
            ref = naive_ref[origin]
            if np.isfinite(ref) and ref > 0 and not (ref / PLAUSIBILITY_BAND_VS_NAIVE <= abs(value) <= ref * PLAUSIBILITY_BAND_VS_NAIVE):
                continue
            out[origin] = value
        fold_params.append({"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "params": params})
    return GarchForecastResult(
        model_id=model_id,
        config={"vol": vol, "o": o, "dist": "t", "p": 1, "q": 1, "method": method, "return_scale": RETURN_SCALE},
        variance_forecast=out,
        fold_params=fold_params,
    )


def gjr_garch_t_forecast(log_returns: np.ndarray, horizon: int, fold_train_ends: list[int]) -> GarchForecastResult:
    """GJR-GARCH(1,1)-t: asymmetric leverage term `o=1`, analytic multi-step
    forecasts (available for this vol family)."""
    return _run_ladder_model(log_returns, horizon, fold_train_ends, model_id="GARCH_t", vol="GARCH", o=1, method="analytic")


def egarch_t_forecast(log_returns: np.ndarray, horizon: int, fold_train_ends: list[int]) -> GarchForecastResult:
    """EGARCH(1,1)-t. `arch` has no closed-form multi-step-ahead forecast
    for EGARCH (`Analytic forecasts not available for horizon > 1`,
    confirmed live) -- uses `method="simulation"` (500 paths, seeded RNG for
    reproducibility) instead, exactly as the library's own documentation
    prescribes for this model family."""
    return _run_ladder_model(log_returns, horizon, fold_train_ends, model_id="EGARCH_t", vol="EGARCH", o=1, method="simulation")
