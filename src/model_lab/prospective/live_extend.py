"""Builds an in-memory `ReturnSeries` that extends the frozen dataset with
REAL, live-fetched bars beyond its own last known date -- never written
back to the frozen `DatasetSnapshot`/registry, never read by any other
Model Lab code path. Exists so a SECOND (and any later) round of
prospective capture can use the previous round's own resolved outcome as
new information, exactly as a genuinely live, day-over-day pipeline would,
without this project's static research dataset ever being mutated.

Only `closes` carries real information for every forecasting function this
candidate uses (`vol/baselines.py`, `vol/garch_models.py` read only
`log_returns`, derived from `closes`) -- confirmed by inspection, not
assumed: `highs`/`opens`/`lows` are never read by any function in this
candidate's own code path. Extended rows therefore set
`highs=lows=opens=close` as an explicit placeholder, clearly documented
here rather than silently passed off as real intraday data."""

from __future__ import annotations

import numpy as np

from model_lab.prospective.live_fetch import fetch_bars_after
from model_lab.vol.data import ReturnSeries


def extend_series_with_live_bars(series: ReturnSeries, *, lookahead_days: int = 10) -> ReturnSeries | None:
    """Returns a NEW `ReturnSeries` (same ticker/dataset_id/data_fingerprint
    -- this is an in-memory extension of the same logical series, not a
    different dataset) with every real, live bar found strictly after
    `series.dates[-1]` appended, or `None` if no new bar exists yet (the
    caller must treat that as "cannot extend right now", never fabricate
    one)."""
    new_bars = fetch_bars_after(series.ticker, series.dates[-1], lookahead_days=lookahead_days)
    if not new_bars:
        return None
    new_dates = list(series.dates)
    new_closes = list(series.closes)
    new_log_returns = list(series.log_returns)
    for bar in new_bars:
        prev_close = new_closes[-1]
        close = float(bar.close)
        new_dates.append(bar.trading_date)
        new_closes.append(close)
        new_log_returns.append(float(np.log(close / prev_close)))
    closes_arr = np.array(new_closes)
    return ReturnSeries(
        ticker=series.ticker,
        dataset_id=series.dataset_id,
        dates=new_dates,
        closes=closes_arr,
        highs=closes_arr,  # placeholder -- see module docstring; never read by this candidate's code path
        lows=closes_arr,
        opens=closes_arr,
        log_returns=np.array(new_log_returns),
        data_fingerprint=series.data_fingerprint,
    )
