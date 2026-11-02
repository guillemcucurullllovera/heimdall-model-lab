from __future__ import annotations

from model_lab.splitter import (
    LabelSpan,
    embargo_indices,
    purge_and_embargo,
    purge_indices,
    spans_overlap,
)


def test_no_overlap_far_apart() -> None:
    train = LabelSpan(0, 0.0, 1.0)
    test = LabelSpan(1, 10.0, 11.0)
    assert not spans_overlap(train, test)


def test_condition1_train_start_within_test() -> None:
    train = LabelSpan(0, 5.5, 12.0)
    test = LabelSpan(1, 5.0, 6.0)
    assert spans_overlap(train, test)


def test_condition2_train_end_within_test() -> None:
    train = LabelSpan(0, 0.0, 5.5)
    test = LabelSpan(1, 5.0, 6.0)
    assert spans_overlap(train, test)


def test_condition3_train_envelops_test() -> None:
    train = LabelSpan(0, 0.0, 10.0)
    test = LabelSpan(1, 4.0, 6.0)
    assert spans_overlap(train, test)


def test_purge_indices_removes_only_overlapping() -> None:
    train_spans = [LabelSpan(0, 0.0, 1.0), LabelSpan(1, 5.0, 5.5), LabelSpan(2, 20.0, 21.0)]
    test_spans = [LabelSpan(99, 5.0, 6.0)]
    purged = purge_indices(train_spans, test_spans)
    assert purged == {1}


def test_embargo_removes_observations_shortly_after_test_block() -> None:
    train_spans = [LabelSpan(0, 6.2, 6.5), LabelSpan(1, 8.0, 8.5)]
    test_blocks = [(5.0, 6.0)]
    purged = embargo_indices(train_spans, test_blocks, embargo_length=1.0)
    assert purged == {0}  # 6.2 is within (6.0, 7.0]; 8.0 is not


def test_nested_purge_includes_outer_test_set() -> None:
    train_spans = [LabelSpan(0, 100.0, 101.0)]
    inner_test = [LabelSpan(1, 50.0, 51.0)]
    outer_test = [LabelSpan(2, 100.0, 100.5)]
    purged = purge_and_embargo(train_spans, inner_test, [], 0.0, outer_test_spans=outer_test)
    assert 0 in purged  # only purged because of the OUTER test set overlap
