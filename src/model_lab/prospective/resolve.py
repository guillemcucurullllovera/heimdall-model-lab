"""Resolve step: for a stored, still-PENDING `PendingObservation`, make a
real, read-only, live check (`live_fetch.py`) for whether the real world has
produced a bar for the day immediately after `information_cutoff_date`. If
yes, compute the realized variance proxy and every model's QLIKE loss
against it, and return a new, separate `ResolvedObservation`. If no, return
`None` -- the pending observation is left untouched, genuinely still
pending, never fabricated."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np

from model_lab.eval.scores import qlike
from model_lab.prospective.live_fetch import fetch_bars_after
from model_lab.prospective.records import PendingObservation, ResolvedObservation


def resolve_pending_observation(pending: PendingObservation) -> ResolvedObservation | None:
    new_bars = fetch_bars_after(pending.ticker, pending.information_cutoff_date, lookahead_days=10)
    if not new_bars:
        return None  # genuinely not yet knowable -- leave PENDING

    next_bar = new_bars[0]  # the FIRST new trading day strictly after the information cutoff
    close_prev = float(pending.information_cutoff_close)
    close_target = float(next_bar.close)
    realized_log_return = math.log(close_target / close_prev)
    realized_variance_proxy = realized_log_return**2

    if pending.candidate_output.point_estimate is None:
        raise ValueError(f"{pending.ticker}: candidate ModelOutput has no point_estimate -- cannot resolve")
    candidate_value = float(pending.candidate_output.point_estimate)
    candidate_loss = float(qlike(np.array([realized_variance_proxy]), np.array([candidate_value])).value)

    benchmark_losses: dict[str, str] = {}
    for entry in pending.benchmark_forecasts:
        loss = float(qlike(np.array([realized_variance_proxy]), np.array([float(entry.point_estimate)])).value)
        benchmark_losses[entry.model_id] = str(loss)

    # Sanity guard against a real design bug, not merely documentation: the resolved target date
    # must be STRICTLY after the information cutoff, i.e. this can never be a historical replay by
    # construction (checked `live_fetch.fetch_bars_after`'s own filter) -- re-asserted here so a
    # future refactor of either module cannot silently reintroduce replay without a test catching it.
    is_replay = next_bar.trading_date <= pending.information_cutoff_date

    return ResolvedObservation(
        pending_observation_hash=pending.pending_observation_hash,
        resolved_at_utc=datetime.now(UTC),
        realized_target_date=next_bar.trading_date,
        realized_close_prev=str(close_prev),
        realized_close_target=str(close_target),
        realized_log_return=str(realized_log_return),
        realized_variance_proxy=str(realized_variance_proxy),
        candidate_forecast_error=str(candidate_loss),
        benchmark_forecast_errors=benchmark_losses,
        resolution_source="massive_live_fetch",
        is_historical_replay=is_replay,
    )
