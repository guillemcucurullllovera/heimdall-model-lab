"""MODEL-1 canonical splitter. See MODEL1_foundation_contracts.md §13."""

from model_lab.splitter.cpcv import CpcvPath, CpcvResult, CpcvSplit, build_cpcv, n_paths, n_splits
from model_lab.splitter.expanding import WalkForwardFold, build_expanding_walk_forward
from model_lab.splitter.purge_embargo import (
    LabelSpan,
    embargo_indices,
    purge_and_embargo,
    purge_indices,
    spans_overlap,
)

__all__ = [
    "LabelSpan",
    "spans_overlap",
    "purge_indices",
    "embargo_indices",
    "purge_and_embargo",
    "build_cpcv",
    "CpcvResult",
    "CpcvSplit",
    "CpcvPath",
    "n_splits",
    "n_paths",
    "build_expanding_walk_forward",
    "WalkForwardFold",
]
