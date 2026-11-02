"""Combinatorial Purged Cross-Validation — MODEL1_foundation_contracts.md §13.

"N ordered groups, k test groups, C(N,k) splits and phi[N,k] = (k/N)*C(N,k)
paths (worked example N=6,k=2 -> 15 splits, 5 paths); bookkeeping that
assigns each test group to exactly one path."

Path construction: every group serves as a test group in exactly phi splits
(a combinatorial identity: a group appears in C(N-1,k-1) = C(N,k)*k/N = phi
of the C(N,k) k-combinations). For each group, its phi test-appearances are
given a canonical rank 0..phi-1 (by split generation order); path `p` is
defined as: for every group, use the split ranked `p` among that group's
test-appearances as the out-of-sample source for that group. This is a
bijection between {(group, split) : group in test(split)} and {(group,
path)} (both have size N*phi = C(N,k)*k), so every group appears in exactly
phi paths (trivially, once each) and every (group,split) test-membership is
used by exactly one path -- satisfying the required path-count and
per-group-uniformity properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from model_lab.splitter.purge_embargo import LabelSpan, purge_and_embargo


def n_splits(N: int, k: int) -> int:
    return math.comb(N, k)


def n_paths(N: int, k: int) -> int:
    """phi[N,k] = (k/N) * C(N,k). The worked example N=6,k=2 gives
    (2/6)*15 = 5."""
    return (k * math.comb(N, k)) // N


@dataclass(frozen=True)
class CpcvSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_groups: tuple[int, ...]
    train_indices: frozenset[int]
    test_indices: frozenset[int]


@dataclass(frozen=True)
class CpcvPath:
    path_id: int
    # group_id -> the split_id whose prediction for that group is used on this path
    group_to_split: dict[int, int]


@dataclass(frozen=True)
class CpcvResult:
    N: int
    k: int
    splits: list[CpcvSplit]
    paths: list[CpcvPath]


def build_cpcv(
    groups: list[list[LabelSpan]],
    N: int,
    k: int,
    *,
    embargo_length: float = 0.0,
) -> CpcvResult:
    """`groups[i]` is the list of LabelSpans belonging to ordered group i
    (0..N-1); `len(groups) must == N`."""
    if len(groups) != N:
        raise ValueError(f"expected {N} groups, got {len(groups)}")

    all_combos = list(combinations(range(N), k))  # C(N,k) splits, in canonical order
    splits: list[CpcvSplit] = []
    # For path bookkeeping: for each group, the ordered list of split_ids in
    # which it is a test group (in split-generation order == canonical rank).
    group_test_splits: dict[int, list[int]] = {g: [] for g in range(N)}

    for split_id, test_group_ids in enumerate(all_combos):
        train_group_ids = tuple(g for g in range(N) if g not in test_group_ids)
        test_spans = [span for g in test_group_ids for span in groups[g]]
        train_spans = [span for g in train_group_ids for span in groups[g]]
        # test_blocks: contiguous [min(t_start), max(t_end)] per test group,
        # for the embargo rule.
        test_blocks = []
        for g in test_group_ids:
            spans = groups[g]
            if spans:
                test_blocks.append((min(s.t_start for s in spans), max(s.t_end for s in spans)))
        purged = purge_and_embargo(train_spans, test_spans, test_blocks, embargo_length)
        train_indices = frozenset(s.index for s in train_spans if s.index not in purged)
        test_indices = frozenset(s.index for s in test_spans)
        splits.append(
            CpcvSplit(
                split_id=split_id,
                test_groups=test_group_ids,
                train_groups=train_group_ids,
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
        for g in test_group_ids:
            group_test_splits[g].append(split_id)

    phi = n_paths(N, k)
    for g in range(N):
        if len(group_test_splits[g]) != phi:
            raise AssertionError(
                f"group {g} appears as a test group in {len(group_test_splits[g])} splits, expected phi={phi} "
                "(combinatorial identity C(N-1,k-1) == phi violated -- indicates a bug, not a data issue)"
            )

    paths: list[CpcvPath] = []
    for p in range(phi):
        mapping = {g: group_test_splits[g][p] for g in range(N)}
        paths.append(CpcvPath(path_id=p, group_to_split=mapping))

    return CpcvResult(N=N, k=k, splits=splits, paths=paths)
