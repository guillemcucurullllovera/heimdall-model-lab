"""Risk-reduction decision-utility check -- the appropriate evaluation
for anomaly/changepoint DETECTORS, as opposed to a forced P&L backtest.

Question asked: if a risk manager cut exposure to zero on every day this
detector flags (top-decile score, matching the SAME threshold convention
`anomaly.evaluate`/`changepoint.evaluate` already use for their own
illustrative-date checks), what real forward-return impact would that
have had? This is a risk-AVOIDANCE utility check, not an alpha claim --
it asks nothing about whether unflagged days are profitable to hold, only
whether flagged days were disproportionately bad.

Same descriptive-threshold convention and caveat as the existing
illustrative-date checks this reuses: the top-decile threshold is
computed over the FULL (non-causal) OOS score distribution, exactly like
`anomaly.evaluate._illustrative_check` already does -- a genuinely causal
(expanding-window) threshold would be a reasonable future refinement, not
built here.
"""

from __future__ import annotations

import numpy as np

from model_lab.contracts.economic_evaluation import RiskReductionUtilityResult

TOP_DECILE_THRESHOLD = 0.90


def anomaly_risk_reduction_utility(
    model_id: str, scores: np.ndarray, panel_returns: np.ndarray, oos_indices: list[int]
) -> RiskReductionUtilityResult | None:
    """`panel_returns` is the equal-weight six-ETF panel's own daily log
    return (the natural "what would a diversified holder experience"
    quantity for a cross-sectional anomaly score, which is itself
    cross-sectional, not single-ticker)."""
    valid = [i for i in oos_indices if not np.isnan(scores[i]) and i + 1 < len(panel_returns)]
    if len(valid) < 30:
        return None
    idx = np.array(valid)
    valid_scores = scores[idx]
    threshold_value = np.percentile(valid_scores, TOP_DECILE_THRESHOLD * 100)
    flagged = valid_scores >= threshold_value
    forward_returns = panel_returns[idx + 1]

    n_flagged = int(flagged.sum())
    n_unflagged = int((~flagged).sum())
    if n_flagged < 5 or n_unflagged < 5:
        return None
    mean_flagged = float(np.mean(forward_returns[flagged]))
    mean_unflagged = float(np.mean(forward_returns[~flagged]))
    avoided_loss = float(np.sum(np.clip(forward_returns[flagged], None, 0.0)))

    return RiskReductionUtilityResult(
        model_id=model_id,
        mean_forward_return_flagged_days=str(mean_flagged),
        mean_forward_return_unflagged_days=str(mean_unflagged),
        n_flagged=n_flagged, n_unflagged=n_unflagged,
        hypothetical_avoided_loss_if_flat_on_flagged_days=str(avoided_loss),
        interpretation=(
            f"On the {n_flagged} days {model_id} flags in its own top decile, the equal-weight "
            f"six-ETF panel's mean next-day log return was {mean_flagged:.5f}, vs {mean_unflagged:.5f} "
            f"on the {n_unflagged} unflagged days. "
            + ("Flagged days are worse on average -- consistent with a risk-avoidance use."
               if mean_flagged < mean_unflagged else
               "Flagged days are NOT worse on average in this sample -- no risk-avoidance value demonstrated here.")
        ),
    )


def changepoint_risk_reduction_utility(
    model_id: str, alarm_or_prob: np.ndarray, log_returns: np.ndarray, oos_indices: list[int], *, is_probability: bool
) -> RiskReductionUtilityResult | None:
    valid = [i for i in oos_indices if i + 1 < len(log_returns) and not np.isnan(alarm_or_prob[i])]
    if len(valid) < 30:
        return None
    idx = np.array(valid)
    values = alarm_or_prob[idx]
    if is_probability:
        threshold_value = np.percentile(values, TOP_DECILE_THRESHOLD * 100)
        flagged = values >= threshold_value
    else:
        flagged = values.astype(bool)
    forward_returns = log_returns[idx + 1]

    n_flagged = int(flagged.sum())
    n_unflagged = int((~flagged).sum())
    if n_flagged < 5 or n_unflagged < 5:
        return None
    mean_flagged = float(np.mean(forward_returns[flagged]))
    mean_unflagged = float(np.mean(forward_returns[~flagged]))
    avoided_loss = float(np.sum(np.clip(forward_returns[flagged], None, 0.0)))

    return RiskReductionUtilityResult(
        model_id=model_id,
        mean_forward_return_flagged_days=str(mean_flagged),
        mean_forward_return_unflagged_days=str(mean_unflagged),
        n_flagged=n_flagged, n_unflagged=n_unflagged,
        hypothetical_avoided_loss_if_flat_on_flagged_days=str(avoided_loss),
        interpretation=(
            f"On the {n_flagged} days {model_id} flags, mean next-day log return was "
            f"{mean_flagged:.5f}, vs {mean_unflagged:.5f} on {n_unflagged} unflagged days. "
            + ("Flagged days are worse on average -- consistent with a risk-avoidance use."
               if mean_flagged < mean_unflagged else
               "Flagged days are NOT worse on average in this sample -- no risk-avoidance value demonstrated here.")
        ),
    )
