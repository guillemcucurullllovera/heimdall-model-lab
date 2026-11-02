"""Builds an aligned cross-sectional daily log-return panel across multiple
tickers (inner-joined on date -- a day missing for any one ticker is
dropped for all, so every row is a genuinely simultaneous observation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.vol.data import load_return_series


@dataclass(frozen=True)
class ReturnPanel:
    tickers: list[str]
    dates: list[str]  # aligned to returns' rows
    returns: np.ndarray  # shape (n_dates, n_tickers)
    data_fingerprints: dict[str, str]
    # Remediation (independent review finding P2-5): per-ticker dataset_id
    # (distinct from data_fingerprints' content hash), needed so a
    # panel-level shadow record (anomaly's cross-sectional Mahalanobis
    # scorer -- the only model family here whose unit of analysis is the
    # panel, not one ticker) can populate ModelVersion.data_snapshot_ids
    # honestly, the same way every per-ticker shadow builder already uses
    # ReturnSeries.dataset_id.
    dataset_ids: dict[str, str]


def build_return_panel(
    tickers: list[str], registry: DatasetSnapshotRegistry, *, dataset_id_map: dict[str, str] | None = None
) -> ReturnPanel:
    per_ticker = {t: load_return_series(t, registry, dataset_id_map=dataset_id_map) for t in tickers}
    date_sets = [set(s.dates[1:]) for s in per_ticker.values()]
    common_dates = sorted(set.intersection(*date_sets))
    if not common_dates:
        raise ValueError(f"no common dates across tickers: {tickers}")

    returns = np.empty((len(common_dates), len(tickers)), dtype=float)
    for j, t in enumerate(tickers):
        series = per_ticker[t]
        date_to_ret = dict(zip(series.dates[1:], series.log_returns, strict=True))
        returns[:, j] = [date_to_ret[d] for d in common_dates]

    return ReturnPanel(
        tickers=list(tickers),
        dates=common_dates,
        returns=returns,
        data_fingerprints={t: s.data_fingerprint for t, s in per_ticker.items()},
        dataset_ids={t: s.dataset_id for t, s in per_ticker.items()},
    )
