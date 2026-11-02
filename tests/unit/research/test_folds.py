"""Remediation (2026-10-02 Phase 2, independent review finding P2-8):
`research/folds.py::expanding_fold_train_ends` discards the real purged
`train_indices` that `build_expanding_walk_forward` computes internally,
and six separate model families (`vol/garch_models.py`,
`tail/unconditional.py`, `tail/conditional.py`, `regime/markov_switching.py`,
`anomaly/covariance_models.py`, `anomaly/isolation_forest_model.py`) each
independently re-derive the purged fit boundary as the manual shortcut
`fit_end_idx = max(train_end_idx - purge_days, FLOOR)`, relying only on a
comment-only claim ("verified during remediation") that this is
algebraically identical to what `purge_and_embargo`/`purge_indices`
actually computes for this pure-expanding, contiguous-test-block fold
topology. No test in the repository called both and compared them.

This module adds that missing equivalence test: for the REAL production
construction `expanding_fold_train_ends` uses internally (`build_year_
groups`, `LabelSpan`, `build_expanding_walk_forward`, `embargo_length=
horizon`), the real purged `train_indices` of every fold actually produced
by that construction equal the contiguous prefix `[0, train_end_idx -
horizon)` the six model families' shortcut assumes -- across every horizon
those families actually use (1, 5, 21, 45) and across fold-count/length
variations, not just one hand-picked case.
"""

from __future__ import annotations

import pytest

from model_lab.research.folds import build_year_groups
from model_lab.splitter.expanding import build_expanding_walk_forward
from model_lab.splitter.purge_embargo import LabelSpan


def _synthetic_business_dates(n_years: int, start_year: int = 2016) -> list[str]:
    """A simple, deterministic weekday-only calendar -- not a real trading
    calendar, but sufficient to build multi-year groups of realistic size
    (~250/year), which is what matters for this equivalence check (whether
    the floor-clipping edge case in `max(train_end_idx - purge_days,
    FLOOR)` ever binds for production-realistic fold sizes)."""
    import datetime as _dt

    dates: list[str] = []
    d = _dt.date(start_year, 1, 1)
    end = _dt.date(start_year + n_years, 1, 1)
    while d < end:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return dates


def _real_purged_train_indices_per_fold(dates: list[str], horizon: int, min_train_groups: int) -> list[tuple[int, frozenset[int]]]:
    """Reconstructs EXACTLY what `expanding_fold_train_ends` builds
    internally (same `build_year_groups`/`LabelSpan`/`build_expanding_
    walk_forward` call, same `embargo_length=float(horizon)`), but -- unlike
    that function -- keeps the real, purged `train_indices` each fold's own
    `WalkForwardFold` computes, which `expanding_fold_train_ends` itself
    discards (see its own docstring's remediation note for why, and this
    test for the equivalence claim that justifies discarding it)."""
    groups_idx = build_year_groups(dates)
    groups = [[LabelSpan(index=i, t_start=float(i), t_end=float(i + horizon)) for i in g] for g in groups_idx]
    folds = build_expanding_walk_forward(groups, min_train_groups=min_train_groups, embargo_length=float(horizon))
    out = []
    for f in folds:
        if not f.test_indices:
            continue
        train_end_idx = min(f.test_indices)
        out.append((train_end_idx, f.train_indices))
    return out


@pytest.mark.parametrize("horizon", [1, 5, 21, 45])
@pytest.mark.parametrize("n_years", [5, 9])
def test_real_purge_equals_six_model_families_shortcut_for_every_production_fold(horizon: int, n_years: int) -> None:
    """The six model families' shared shortcut is `fit_end_idx =
    train_end_idx - purge_days` with `purge_days` defaulted to `horizon`
    (matching each caller's own usage -- see `research/folds.py`'s own
    docstring). This asserts that shortcut's implied train-index set,
    `set(range(0, train_end_idx - horizon))`, equals the REAL purged
    `train_indices` `purge_and_embargo` computes for every fold actually
    produced by the production construction, at every horizon the vol/
    tail/regime/anomaly families actually use."""
    dates = _synthetic_business_dates(n_years)
    folds = _real_purged_train_indices_per_fold(dates, horizon=horizon, min_train_groups=2)
    assert folds, "synthetic calendar must produce at least one real fold to test anything"

    for train_end_idx, real_train_indices in folds:
        shortcut_train_indices = frozenset(range(0, train_end_idx - horizon))
        assert real_train_indices == shortcut_train_indices, (
            f"train_end_idx={train_end_idx} horizon={horizon}: real purge_and_embargo output "
            f"({len(real_train_indices)} indices) != the six model families' shortcut "
            f"({len(shortcut_train_indices)} indices) -- the P2-8 equivalence claim does not hold "
            "for this fold"
        )


def test_real_purge_is_a_contiguous_prefix_not_a_scattered_set() -> None:
    """A second, independent check of the same claim `research/folds.py`'s
    docstring makes ('never a scattered set'): the real purged train
    indices for a representative multi-year, horizon=21 topology form an
    exact contiguous range starting at 0, not merely a set of the same
    size as one (which the main test above could not distinguish, since
    `set(range(...))` IS contiguous by construction -- this check instead
    verifies the real output directly, independent of the shortcut)."""
    dates = _synthetic_business_dates(n_years=9)
    folds = _real_purged_train_indices_per_fold(dates, horizon=21, min_train_groups=2)
    assert folds
    for train_end_idx, real_train_indices in folds:
        sorted_indices = sorted(real_train_indices)
        if not sorted_indices:
            continue
        assert sorted_indices == list(range(sorted_indices[0], sorted_indices[-1] + 1)), (
            f"train_end_idx={train_end_idx}: real purged train_indices are not a contiguous range"
        )
        assert sorted_indices[0] == 0
        assert sorted_indices[-1] == train_end_idx - 21 - 1
