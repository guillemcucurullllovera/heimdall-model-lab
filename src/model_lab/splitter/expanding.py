"""Expanding-window walk-forward splitter — the deployment-faithful
comparator alongside CPCV (§13)."""

from __future__ import annotations

from dataclasses import dataclass

from model_lab.splitter.purge_embargo import LabelSpan, purge_and_embargo


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_indices: frozenset[int]
    test_indices: frozenset[int]


def build_expanding_walk_forward(
    groups: list[list[LabelSpan]], *, min_train_groups: int = 1, embargo_length: float = 0.0
) -> list[WalkForwardFold]:
    """`groups[i]` = ordered chronological group i's LabelSpans. Fold f uses
    groups[0:min_train_groups+f] as train (expanding) and group
    [min_train_groups+f] as test, purge+embargo applied."""
    N = len(groups)
    folds: list[WalkForwardFold] = []
    fold_id = 0
    for test_group_idx in range(min_train_groups, N):
        train_group_ids = list(range(test_group_idx))
        test_spans = groups[test_group_idx]
        train_spans = [s for g in train_group_ids for s in groups[g]]
        if not test_spans:
            continue
        test_block = [(min(s.t_start for s in test_spans), max(s.t_end for s in test_spans))]
        purged = purge_and_embargo(train_spans, test_spans, test_block, embargo_length)
        train_indices = frozenset(s.index for s in train_spans if s.index not in purged)
        test_indices = frozenset(s.index for s in test_spans)
        folds.append(WalkForwardFold(fold_id=fold_id, train_indices=train_indices, test_indices=test_indices))
        fold_id += 1
    return folds
