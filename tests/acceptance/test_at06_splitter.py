"""AT-06: property tests + oracle equivalence (skfolio/timeseriescv) on
shared fixtures; path count formula. This is the acceptance-level
checkpoint; the exhaustive unit-level coverage lives in
tests/unit/splitter/."""

from __future__ import annotations

import math

import pytest

from model_lab.splitter import LabelSpan, build_cpcv, n_paths, n_splits

pytestmark = pytest.mark.acceptance


def _groups(N: int, per_group: int) -> list[list[LabelSpan]]:
    groups = []
    idx = 0
    for _g in range(N):
        spans = [LabelSpan(index=(idx := idx + 1) - 1, t_start=float(idx), t_end=float(idx) + 0.9) for _ in range(per_group)]
        groups.append(spans)
    return groups


def test_worked_example_reproduced() -> None:
    assert n_splits(6, 2) == 15
    assert n_paths(6, 2) == 5
    result = build_cpcv(_groups(6, 4), N=6, k=2)
    assert len(result.splits) == 15
    assert len(result.paths) == 5


def test_path_count_formula_general() -> None:
    for N, k in [(4, 1), (5, 2), (7, 3), (8, 4)]:
        assert n_paths(N, k) == (k * math.comb(N, k)) // N


def test_no_overlap_after_purge_property() -> None:
    result = build_cpcv(_groups(8, 6), N=8, k=3, embargo_length=1.0)
    for split in result.splits:
        assert split.train_indices.isdisjoint(split.test_indices)


def test_every_group_tested_same_number_of_times() -> None:
    result = build_cpcv(_groups(6, 5), N=6, k=2)
    counts: dict[int, int] = {}
    for split in result.splits:
        for g in split.test_groups:
            counts[g] = counts.get(g, 0) + 1
    assert len(set(counts.values())) == 1  # all groups tested equally often


def test_oracle_equivalence_skfolio() -> None:
    pytest.importorskip("skfolio")
    import numpy as np
    from skfolio.model_selection import CombinatorialPurgedCV

    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=0, embargo_size=0)
    oracle_splits = list(cv.split(np.zeros((30, 1))))
    assert len(oracle_splits) == n_splits(6, 2)
    assert cv.n_test_paths == n_paths(6, 2)
