"""MODEL-2 candidate specification — LOCKED (Phase 2 of the post-MODEL-2G
non-review continuation). This module is the single source of truth for
"the candidate" everywhere downstream (shadow records, capstone, promotion
gates) read it from, rather than each consumer re-deriving its own notion of
what the candidate is.

Reconstructed strictly from the already-committed `reports/vol_ladder/*.json`
evidence (MODEL-2G) — nothing here was broadened after seeing results; the
`ELIGIBLE_CELLS`/`REJECTED_CELLS` partition below is exactly MODEL-2G's own
finding (9 cells significantly better than `GARCH_t` at DM p<0.05, 0 cells
significantly worse, 15 cells with no significant difference either way).
Do not add a ticker/horizon to `ELIGIBLE_CELLS` without a new, equally
rigorous walk-forward/DM result backing it — this module is deliberately
not auto-computed from the live JSON precisely so that "the candidate"
cannot silently drift as new evidence is added without a conscious edit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateCell:
    ticker: str
    horizon: int


# MODEL-2G finding: Combo_median_EWMA_GARCH_EGARCH significantly beats GARCH_t (DM p<0.05) on
# exactly these 9 of 24 (ticker, horizon) cells. Full numbers: reports/vol_ladder/*.json,
# reports/MODEL2G_RESEARCH_LAYERS_ABC.md §1.
ELIGIBLE_CELLS: tuple[CandidateCell, ...] = (
    CandidateCell("TLT", 1), CandidateCell("TLT", 5), CandidateCell("TLT", 21), CandidateCell("TLT", 45),
    CandidateCell("IEF", 1), CandidateCell("IEF", 5), CandidateCell("IEF", 21), CandidateCell("IEF", 45),
    CandidateCell("GLD", 1),
)

# The remaining 15 cells: no significant difference from GARCH_t either way (never significantly
# worse in any cell -- see MODEL2G §1). These are explicitly NOT part of the candidate's claimed
# edge; the candidate may still be COMPUTED for them (the combination function is well-defined
# everywhere), but no edge over GARCH_t is asserted there.
REJECTED_CELLS: tuple[CandidateCell, ...] = (
    CandidateCell("SPY", 1), CandidateCell("SPY", 5), CandidateCell("SPY", 21), CandidateCell("SPY", 45),
    CandidateCell("QQQ", 1), CandidateCell("QQQ", 5), CandidateCell("QQQ", 21), CandidateCell("QQQ", 45),
    CandidateCell("GLD", 5), CandidateCell("GLD", 21), CandidateCell("GLD", 45),
    CandidateCell("USO", 1), CandidateCell("USO", 5), CandidateCell("USO", 21), CandidateCell("USO", 45),
)

CANDIDATE_MODEL_ID = "Combo_median_EWMA_GARCH_EGARCH"
CANDIDATE_FAMILY = "median-forecast-combination"
CANDIDATE_CONSTITUENTS = ("EWMA_0.94", "GARCH_t", "EGARCH_t")
CANDIDATE_COMBINATION_RULE = "median_variance_combination"  # vol/combination.py
CANDIDATE_BENCHMARK_SET = ("naive_trailing_rv", "EWMA_0.94", "GARCH_t", "EGARCH_t")
CANDIDATE_STRONGEST_BENCHMARK = "GARCH_t"
CANDIDATE_EVALUATION_METRIC = "QLIKE (proper scoring rule for variance forecasts)"
CANDIDATE_STATISTICAL_TEST = "Diebold-Mariano (HAC-corrected), pairwise vs GARCH_t"
CANDIDATE_TASK_ID = "T-VOL"  # the existing T-VOL task this candidate targets (research/shadow only)

# Information available at forecast time: identical to every other vol-ladder model -- real,
# already-realised daily OHLC closes up to and including the forecast origin, nothing from the
# future relative to that origin. The combination itself performs NO estimation of its own (see
# vol/combination.py); it is a fixed, data-independent function of its three members' own
# already-causal forecasts. This is why the candidate requires no NEW leakage proof beyond what
# EWMA_0.94/GARCH_t/EGARCH_t already individually carry (vol/evaluate.py's own no_leakage_proof).
CANDIDATE_INFORMATION_SET = "daily OHLC closes up to and including the forecast origin (EOD, no intraday data)"

CANDIDATE_TRAINING_PROCEDURE = (
    "No training of its own. EWMA_0.94 is a fixed-decay recursive filter (no fitting). "
    "GARCH_t/EGARCH_t are each refit once per calendar-year expanding fold on "
    "data[:train_end_idx - horizon] (purge=horizon), exactly as in the vol ladder. "
    "The combination takes the elementwise median of the three members' variance forecasts "
    "at each origin -- a fixed, non-adaptive combination rule, not a weight-estimation scheme."
)

CANDIDATE_OUTPUT_DESCRIPTION = (
    "A single point forecast: the median of the three constituent models' cumulative h-day "
    "variance forecast at the origin. No native uncertainty/confidence interval is produced by "
    "the combination itself (median combination collapses the 3 members to one number); "
    "dispersion among the 3 members IS available as a separate diagnostic (see Phase 8's "
    "confidence/abstention research) but is not part of the candidate's own output."
)


def is_eligible_cell(ticker: str, horizon: int) -> bool:
    """True iff (ticker, horizon) is one of the 9 cells where the candidate has a
    significant, already-demonstrated edge over GARCH_t. Any other cell returns False --
    including tickers/horizons never evaluated at all -- by design: eligibility is never
    assumed, only ever demonstrated."""
    return CandidateCell(ticker, horizon) in ELIGIBLE_CELLS
