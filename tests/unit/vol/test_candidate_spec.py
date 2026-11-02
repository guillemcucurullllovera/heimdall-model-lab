from __future__ import annotations

from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.candidate_spec import (
    ELIGIBLE_CELLS,
    REJECTED_CELLS,
    CandidateCell,
    is_eligible_cell,
)

HORIZONS = (1, 5, 21, 45)


def test_eligible_and_rejected_cells_partition_all_24_cells_with_no_overlap() -> None:
    all_cells = {CandidateCell(t, h) for t in SIX_ETFS for h in HORIZONS}
    eligible = set(ELIGIBLE_CELLS)
    rejected = set(REJECTED_CELLS)
    assert eligible & rejected == set(), "a cell cannot be both eligible and rejected"
    assert eligible | rejected == all_cells, "every one of the 24 ladder cells must be classified"
    assert len(eligible) == 9
    assert len(rejected) == 15


def test_is_eligible_cell_matches_the_locked_set() -> None:
    assert is_eligible_cell("TLT", 1) is True
    assert is_eligible_cell("IEF", 45) is True
    assert is_eligible_cell("GLD", 1) is True


def test_is_eligible_cell_is_false_for_rejected_and_unknown_combinations() -> None:
    assert is_eligible_cell("SPY", 1) is False
    assert is_eligible_cell("GLD", 21) is False
    assert is_eligible_cell("SPY", 999) is False  # never evaluated at all -- not assumed eligible
    assert is_eligible_cell("NOTATICKER", 1) is False
