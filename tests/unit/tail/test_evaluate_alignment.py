"""Regression test for a real off-by-one bug caught during MODEL-2A
development: `run_tail_ladder` originally compared the VaR/ES forecast
STORED at index i (which predicts the loss on r[i+1], matching
`vol.garch_models`' own forecast-origin convention) against `r[i]` --
the return ALREADY KNOWN at the moment that forecast was made -- instead of
against `r[i+1]`, the day the forecast actually covers. Discovered by a live
sanity check against real SPY data (the conditional GJR-t-filtered-GPD-FHS
model showed an implausible near-zero violation rate at the 97.5%/99%
levels); fixed in `tail/evaluate.py` by indexing the realised return/loss
at `idx_arr + 1`.

This test discriminates the fixed behaviour from the pre-fix bug: it plants
a deliberately WIDE forecast at exactly the CORRECT origin for a single
large loss and a deliberately TIGHT forecast everywhere else. Under correct
(i -> i+1) alignment, the wide forecast absorbs the large loss (zero
violations); under the old (i -> i) bug, the large loss gets paired with a
DIFFERENT origin's tight forecast, producing a spurious violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from model_lab.tail import evaluate as tail_evaluate
from model_lab.vol.data import ReturnSeries


@dataclass(frozen=True)
class _FakeModelResult:
    model_id: str
    config: dict
    var_forecast: dict
    es_forecast: dict
    fold_fits: list


def _business_dates(n: int, start: date) -> list[str]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _build_series(n_per_year: int = 40) -> tuple[ReturnSeries, int]:
    """Two calendar years of business-day dates; returns are tiny (0.0001)
    everywhere except a single large loss planted partway through year 2
    (the only OOS test fold at min_train_groups=1)."""
    dates_y1 = _business_dates(n_per_year, date(2022, 1, 3))
    dates_y2 = _business_dates(n_per_year, date(2023, 1, 2))
    dates = ["2021-12-31", *dates_y1, *dates_y2]  # dates[0] is the "closes[0]" anchor, dates[1:] = return dates
    n = len(dates) - 1
    r = np.full(n, 1e-6)
    shock_idx = n_per_year + n_per_year // 2  # well inside year 2's test fold
    r[shock_idx] = -0.5
    closes = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(r))])
    series = ReturnSeries(
        ticker="FAKE",
        dataset_id="fake_daily_raw",
        dates=dates,
        closes=closes,
        highs=closes,
        lows=closes,
        opens=closes,
        log_returns=r,
        data_fingerprint="f" * 64,
    )
    return series, shock_idx


def _patch_models(monkeypatch: pytest.MonkeyPatch, n: int, wide_origin: int, tail_levels: tuple[float, ...]) -> None:
    def _fake(model_id: str) -> _FakeModelResult:
        var = {p: np.full(n, 0.001) for p in tail_levels}
        es = {p: np.full(n, 0.002) for p in tail_levels}
        for p in tail_levels:
            var[p][wide_origin] = 100.0
            es[p][wide_origin] = 200.0
        return _FakeModelResult(model_id=model_id, config={}, var_forecast=var, es_forecast=es, fold_fits=[])

    monkeypatch.setattr(tail_evaluate, "unconditional_gpd_pot_forecast", lambda *a, **k: _fake("unconditional_GPD_POT"))
    monkeypatch.setattr(tail_evaluate, "gjr_t_filtered_gpd_fhs_forecast", lambda *a, **k: _fake("GJR_t_filtered_GPD_FHS"))


def test_correct_alignment_absorbs_the_shock_with_the_matching_origins_wide_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    series, shock_idx = _build_series()
    n = len(series.log_returns)
    correct_origin = shock_idx - 1  # forecast AT (shock_idx - 1) covers r[shock_idx] under i -> i+1 alignment
    _patch_models(monkeypatch, n, correct_origin, tail_levels=(0.05,))

    result = tail_evaluate.run_tail_ladder(series, min_train_groups=1, tail_levels=(0.05,))

    for model_id in ("unconditional_GPD_POT", "GJR_t_filtered_GPD_FHS"):
        n_violations = result.results[model_id]["0.05"]["n_violations"]
        assert n_violations == 0, (
            f"{model_id}: expected the wide forecast at the CORRECT (shock-1) origin to absorb the planted "
            f"shock under i -> i+1 alignment, got {n_violations} violation(s) -- alignment regressed"
        )


def test_wrong_origin_wide_forecast_does_not_absorb_the_shock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity control proving the test above is discriminating, not
    vacuous: a wide forecast at the WRONG origin (the pre-fix bug's
    pairing) must NOT absorb the shock."""
    series, shock_idx = _build_series()
    n = len(series.log_returns)
    wrong_origin = shock_idx  # the pre-fix (buggy) pairing would compare forecast[shock_idx] against r[shock_idx]
    _patch_models(monkeypatch, n, wrong_origin, tail_levels=(0.05,))

    result = tail_evaluate.run_tail_ladder(series, min_train_groups=1, tail_levels=(0.05,))

    for model_id in ("unconditional_GPD_POT", "GJR_t_filtered_GPD_FHS"):
        n_violations = result.results[model_id]["0.05"]["n_violations"]
        assert n_violations == 1, (
            f"{model_id}: expected exactly 1 violation (the shock, unabsorbed by a wide forecast at the "
            f"wrong origin), got {n_violations}"
        )


def _build_three_year_series(n_per_year: int = 40) -> ReturnSeries:
    """Three calendar years -> two test folds at min_train_groups=1, so a
    real fold-boundary seam exists between them."""
    dates_y1 = _business_dates(n_per_year, date(2021, 1, 4))
    dates_y2 = _business_dates(n_per_year, date(2022, 1, 3))
    dates_y3 = _business_dates(n_per_year, date(2023, 1, 2))
    dates = ["2020-12-31", *dates_y1, *dates_y2, *dates_y3]
    n = len(dates) - 1
    r = np.full(n, 1e-6)
    closes = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(r))])
    return ReturnSeries(
        ticker="FAKE3Y", dataset_id="fake3y_raw", dates=dates, closes=closes, highs=closes,
        lows=closes, opens=closes, log_returns=r, data_fingerprint="g" * 64,
    )


def test_fold_boundary_seam_origin_is_excluded_from_scoring_not_misattributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remediation (independent adversarial review, finding P3-9): the
    ORIGIN whose target (origin+1) is the very first day of a NEW fold's
    test window must be excluded from scoring entirely -- not scored
    against fold k's (older) estimate, which would misattribute a day that
    is really in fold k+1's nominal test year. This test plants a huge loss
    exactly on that seam TARGET day with a WIDE forecast at the seam ORIGIN
    (the pre-fix behaviour would have scored this pair and recorded zero
    violations, silently crediting fold k's estimate for a fold k+1 day);
    the fix must show this point simply absent from `n_oos`, not silently
    "successfully" absorbed."""
    from model_lab.research.folds import expanding_fold_train_ends

    series = _build_three_year_series()
    dates_for_returns = series.dates[1:]
    train_ends, _ = expanding_fold_train_ends(dates_for_returns, horizon=1, min_train_groups=1)
    assert len(train_ends) >= 2, "need at least 2 folds for a real seam to exist"
    seam_origin = train_ends[1] - 1  # last origin of fold 0, whose target is fold 1's first day

    n = len(series.log_returns)
    r = series.log_returns.copy()
    r[seam_origin + 1] = -0.9  # huge loss on the SEAM TARGET day itself
    series = ReturnSeries(
        ticker=series.ticker, dataset_id=series.dataset_id, dates=series.dates, closes=series.closes,
        highs=series.highs, lows=series.lows, opens=series.opens, log_returns=r,
        data_fingerprint=series.data_fingerprint,
    )
    _patch_models(monkeypatch, n, seam_origin, tail_levels=(0.05,))  # wide forecast AT the seam origin

    result = tail_evaluate.run_tail_ladder(series, min_train_groups=1, tail_levels=(0.05,))
    # Exact expected count, independently derived: 2 folds x 40 calendar test
    # days = 80 nominal OOS points, minus 1 for the i<n-1 boundary exclusion,
    # minus 2 for the two real fold-boundary seams (one at each fold's own
    # start) = 78. A regression that stops excluding seams would show 79 or
    # 80 here instead.
    for model_id in ("unconditional_GPD_POT", "GJR_t_filtered_GPD_FHS"):
        level = result.results[model_id]["0.05"]
        assert level["n_oos"] == 78, f"{model_id}: expected exactly 78 OOS points after seam exclusion, got {level['n_oos']}"
