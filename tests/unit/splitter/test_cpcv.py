from __future__ import annotations

import pytest

from model_lab.splitter import LabelSpan, build_cpcv, n_paths, n_splits


def _make_groups(n_groups: int, per_group: int) -> list[list[LabelSpan]]:
    groups: list[list[LabelSpan]] = []
    idx = 0
    for _g in range(n_groups):
        spans = []
        for _ in range(per_group):
            t0 = float(idx)
            spans.append(LabelSpan(index=idx, t_start=t0, t_end=t0 + 0.5))
            idx += 1
        groups.append(spans)
    return groups


def test_worked_example_n6_k2_gives_15_splits_5_paths() -> None:
    assert n_splits(6, 2) == 15
    assert n_paths(6, 2) == 5

    groups = _make_groups(6, 4)
    result = build_cpcv(groups, N=6, k=2)
    assert len(result.splits) == 15
    assert len(result.paths) == 5


def test_every_group_is_a_test_group_in_exactly_phi_splits() -> None:
    groups = _make_groups(6, 4)
    result = build_cpcv(groups, N=6, k=2)
    phi = n_paths(6, 2)
    counts = {g: 0 for g in range(6)}
    for split in result.splits:
        for g in split.test_groups:
            counts[g] += 1
    assert all(c == phi for c in counts.values())


def test_every_path_covers_every_group_exactly_once() -> None:
    groups = _make_groups(6, 4)
    result = build_cpcv(groups, N=6, k=2)
    for path in result.paths:
        assert set(path.group_to_split.keys()) == set(range(6))
        for g, split_id in path.group_to_split.items():
            split = result.splits[split_id]
            assert g in split.test_groups


def test_no_train_test_overlap_after_purge() -> None:
    groups = _make_groups(6, 4)
    result = build_cpcv(groups, N=6, k=2, embargo_length=1.0)
    for split in result.splits:
        assert split.train_indices.isdisjoint(split.test_indices)


@pytest.mark.parametrize("N,k", [(4, 1), (5, 2), (6, 2), (6, 3), (7, 3)])
def test_path_count_formula_various_nk(N: int, k: int) -> None:
    import math

    expected_splits = math.comb(N, k)
    expected_paths = (k * math.comb(N, k)) // N
    groups = _make_groups(N, 3)
    result = build_cpcv(groups, N=N, k=k)
    assert len(result.splits) == expected_splits
    assert len(result.paths) == expected_paths


def test_embargo_monotonicity_never_increases_usable_train_data() -> None:
    groups = _make_groups(6, 5)
    small = build_cpcv(groups, N=6, k=2, embargo_length=0.0)
    large = build_cpcv(groups, N=6, k=2, embargo_length=3.0)
    for s_small, s_large in zip(small.splits, large.splits, strict=True):
        assert s_large.train_indices <= s_small.train_indices
