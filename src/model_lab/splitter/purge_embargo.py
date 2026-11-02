"""Label-span purge + embargo — MODEL1_foundation_contracts.md §13.

"Purge every train observation whose label span overlaps a test label span
(the three sufficient overlap conditions of the AFML §7.4.1 definition --
cited bibliographically, implemented independently); embargo >= max(label
horizon, publication lag) after every test block; nested inner loops purge
against the OUTER test set as well."

The three sufficient overlap conditions (standard result, cited
bibliographically; implemented from first principles, no source code
copied): for a train label span [ts, te] and a test label span [Ts, Te],
the train observation must be purged if ANY of:
  1. ts falls within the test span:      Ts <= ts <= Te
  2. te falls within the test span:      Ts <= te <= Te
  3. the train span envelops the test:   ts <= Ts and te >= Te
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpan:
    """A generic label span. `t_start`/`t_end` may be any totally-ordered,
    subtractable type (float day-offsets, or anything supporting <=, -);
    floats are used throughout this module and its tests for simplicity."""

    index: int
    t_start: float
    t_end: float


def spans_overlap(train: LabelSpan, test: LabelSpan) -> bool:
    cond1 = test.t_start <= train.t_start <= test.t_end
    cond2 = test.t_start <= train.t_end <= test.t_end
    cond3 = train.t_start <= test.t_start and train.t_end >= test.t_end
    return cond1 or cond2 or cond3


def purge_indices(train_spans: list[LabelSpan], test_spans: list[LabelSpan]) -> set[int]:
    """Indices of every train observation whose label span overlaps ANY test
    label span, per the three sufficient conditions above."""
    purged: set[int] = set()
    for tr in train_spans:
        if any(spans_overlap(tr, te) for te in test_spans):
            purged.add(tr.index)
    return purged


def embargo_indices(train_spans: list[LabelSpan], test_blocks: list[tuple[float, float]], embargo_length: float) -> set[int]:
    """Indices of every train observation whose t_start falls within
    (test_end, test_end + embargo_length] for any test block -- i.e. the
    embargo period immediately following each contiguous test block."""
    purged: set[int] = set()
    for tr in train_spans:
        for _, te_end in test_blocks:
            if te_end < tr.t_start <= te_end + embargo_length:
                purged.add(tr.index)
    return purged


def purge_and_embargo(
    train_spans: list[LabelSpan],
    test_spans: list[LabelSpan],
    test_blocks: list[tuple[float, float]],
    embargo_length: float,
    *,
    outer_test_spans: list[LabelSpan] | None = None,
) -> set[int]:
    """Combined purge + embargo. `outer_test_spans`, if given, is ALSO purged
    against (nested inner-loop rule: an inner-loop split must purge against
    the OUTER test set as well as its own inner test set)."""
    purged = purge_indices(train_spans, test_spans)
    purged |= embargo_indices(train_spans, test_blocks, embargo_length)
    if outer_test_spans:
        purged |= purge_indices(train_spans, outer_test_spans)
    return purged
