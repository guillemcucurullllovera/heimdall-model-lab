"""Loads real, already-acquired daily close series from the data estate and
derives log returns. No new data is fetched here -- this module only reads
what MODEL-1/MODEL-2A already acquired and registered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import ETF_PRIMARY_DATASET_ID

# MODEL-2A's own VIX-family ETP universe (scripts/build_vix_etp_universe.py) --
# not part of the frozen T-VOL/T-TAIL instrument universe, kept separate.
VIX_ETP_PRIMARY_DATASET_ID = {
    "VIXY": "vixy_daily_split_adjusted",
    "UVXY": "uvxy_daily_split_adjusted",
    "SVXY": "svxy_daily_split_adjusted",
    "VXX": "vxx_daily_split_adjusted",
}


@dataclass(frozen=True)
class ReturnSeries:
    ticker: str
    dataset_id: str
    dates: list[str]  # ISO date strings, len = len(closes); returns[i] = ln(closes[i]/closes[i-1])
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    opens: np.ndarray
    log_returns: np.ndarray  # length len(dates)-1, aligned to dates[1:]
    data_fingerprint: str  # sha256 logical_content_hash of the underlying DatasetSnapshot


def load_return_series(
    ticker: str, registry: DatasetSnapshotRegistry, *, dataset_id_map: dict[str, str] | None = None
) -> ReturnSeries:
    """Loads the real adjusted daily OHLC series for `ticker` (using the
    SAME primary dataset id the frozen T-VOL/T-TAIL task registry resolves,
    by default -- i.e. the split-adjusted series for USO, raw for the other
    five, exactly per P0-4/`tasks/registry.py::ETF_PRIMARY_DATASET_ID`) and
    derives log returns. Fails closed (raises) if the dataset is not
    registered -- never fabricates a series."""
    mapping = dataset_id_map if dataset_id_map is not None else ETF_PRIMARY_DATASET_ID
    dataset_id = mapping[ticker]
    snapshot = registry.get_snapshot(dataset_id)
    if snapshot is None:
        raise ValueError(f"{ticker}: dataset {dataset_id!r} is not registered in the data estate")
    rows = registry.get_rows(dataset_id)
    if not rows:
        raise ValueError(f"{ticker}: dataset {dataset_id!r} has zero rows")

    ticker_rows = [r for r in rows if r.series_id == ticker]
    ticker_rows.sort(key=lambda r: r.observation_period)
    dates = [r.observation_period.isoformat() for r in ticker_rows]
    bars = [json.loads(r.value) for r in ticker_rows]
    closes = np.array([float(b["c"]) for b in bars], dtype=float)
    highs = np.array([float(b["h"]) for b in bars], dtype=float)
    lows = np.array([float(b["l"]) for b in bars], dtype=float)
    opens = np.array([float(b["o"]) for b in bars], dtype=float)
    log_returns = np.diff(np.log(closes))

    return ReturnSeries(
        ticker=ticker,
        dataset_id=dataset_id,
        dates=dates,
        closes=closes,
        highs=highs,
        lows=lows,
        opens=opens,
        log_returns=log_returns,
        data_fingerprint=snapshot.logical_content_hash,
    )
