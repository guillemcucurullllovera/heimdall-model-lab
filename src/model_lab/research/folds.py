"""Calendar-year expanding walk-forward folds, built on MODEL-1's frozen
`splitter.expanding`/`splitter.purge_embargo` (§13) -- shared by `vol` and
`tail`'s evaluation harnesses so both use IDENTICAL fold boundaries for a
given (dates, horizon, min_train_groups), making cross-ladder comparisons
apples-to-apples.
"""

from __future__ import annotations

from model_lab.splitter.expanding import build_expanding_walk_forward
from model_lab.splitter.purge_embargo import LabelSpan


def build_year_groups(dates: list[str]) -> list[list[int]]:
    """Groups return-index positions by calendar year of `dates[i]`
    (chronologically ordered: group 0 = earliest year)."""
    years: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        year = d[:4]
        years.setdefault(year, []).append(i)
    return [years[y] for y in sorted(years)]


def expanding_fold_train_ends(
    dates: list[str], horizon: int, *, min_train_groups: int = 2
) -> tuple[list[int], list[list[int]]]:
    """Returns (train_end_indices, test_index_lists) -- one pair per fold,
    in chronological order. `train_end_indices[k]` is the first CALENDAR
    test index of fold k (the real fold/forecast-origin boundary, always
    used as-is for scoring and for where forecasts start); `test_index_
    lists[k]` is that fold's real calendar test-set indices.

    ## Purge/embargo -- what this function does and does NOT do

    (Remediation note, independent adversarial review, finding P2-5: an
    earlier version of this docstring claimed `train_end_indices[k]` was
    directly "the number of observations fold k's model may train on" --
    i.e. that purge/embargo was already baked into the returned boundary.
    It was not: `build_expanding_walk_forward`'s own purged `train_indices`
    were computed here and then silently discarded, so every caller was
    actually fitting on the FULL, unpurged calendar-year prefix while every
    phase report described embargo as enforced.)

    This function itself returns only the real calendar boundaries,
    unpurged -- `build_expanding_walk_forward`'s embargo-aware
    `train_indices` are NOT threaded through here. The purge is instead
    applied DOWNSTREAM, explicitly, by each model family's own
    `purge_days` parameter (`vol.garch_models`, `tail.unconditional`,
    `tail.conditional`, `regime.markov_switching`,
    `anomaly.covariance_models`, `anomaly.isolation_forest_model` --
    search each for `purge_days`): every one of them fits parameters on
    `data[:train_end_idx - purge_days]`, not `data[:train_end_idx]`,
    while still using the real, unpurged `train_end_idx` as the forecast
    origin / fold-ownership boundary. `purge_days` defaults to each
    caller's own `horizon` (matching T-VOL/T-TAIL's own `embargo_rule=
    "max(h, publication_lag)"`), confirmed algebraically equivalent to
    `purge_indices`'s own output for this pure-expanding (train strictly
    before test, never after) fold topology -- verified during
    remediation: the real purged `train_indices` this function's own
    `build_expanding_walk_forward` call computes internally is exactly the
    contiguous prefix `[0, train_end_idx - horizon)` for every fold in this
    topology, never a scattered set, so shrinking the fit slice by
    `horizon` downstream reproduces it exactly without needing to thread a
    non-contiguous index set through every model family's fitting code
    (which would not work directly against `arch`/`statsmodels`/`sklearn`,
    all of which expect a plain contiguous array).

    Remediation (2026-10-02 Phase 2, independent review finding P2-8):
    that equivalence was, until this pass, a comment-only claim with no
    automated test calling both `purge_and_embargo` and the six families'
    manual shortcut on the real production construction and comparing
    them -- `tests/unit/research/test_folds.py` now does exactly that,
    parametrized over every horizon (1, 5, 21, 45) the vol/tail ladders
    actually use, so a future change to this fold topology (e.g. adopting
    the non-contiguous CPCV folds `splitter.cpcv` already supports) that
    silently breaks the equivalence will fail a test, not merely go
    unnoticed until a reviewer re-derives it by hand."""
    groups_idx = build_year_groups(dates)
    groups = [[LabelSpan(index=i, t_start=float(i), t_end=float(i + horizon)) for i in g] for g in groups_idx]
    folds = build_expanding_walk_forward(groups, min_train_groups=min_train_groups, embargo_length=float(horizon))
    train_ends: list[int] = []
    test_index_lists: list[list[int]] = []
    for f in folds:
        if not f.test_indices:
            continue
        train_ends.append(min(f.test_indices))
        test_index_lists.append(sorted(f.test_indices))
    return train_ends, test_index_lists
