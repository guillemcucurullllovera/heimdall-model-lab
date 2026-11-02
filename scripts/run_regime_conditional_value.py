#!/usr/bin/env python3
"""Layer A research probe, follow-up to MODEL-2D's negative result: does the
2-state regime blend add value CONDITIONAL on being in a high-uncertainty /
transitional period, even though MODEL-2D found it loses UNCONDITIONALLY
(pooled across all days) to GARCH_t?

Hypothesis: a discrete regime read might be uninformative on "easy" days
(clearly low-vol or clearly high-vol, where GARCH_t's continuous filter
already has it right) but could still add value on "hard" days where the
regime classifier is genuinely uncertain (`high_vol_prob` near 0.5) -- exactly
the days a continuous GARCH recursion might be slow to react to a regime
change. This is a DIFFERENT question from MODEL-2D's average-case comparison,
using the SAME real forecasts and data (no new fitting), just a conditional
(bucketed) re-scoring instead of a pooled one.

Decision rule declared BEFORE looking at results (pre-registration discipline
per instruction): 3 fixed, round-number buckets on |high_vol_prob - 0.5|
(distance from maximal regime uncertainty) -- "confident" (>0.4, i.e.
high_vol_prob outside [0.1, 0.9]), "uncertain" (<=0.1, i.e. high_vol_prob in
[0.4, 0.6]), "transitional" (the remainder). RETAIN the "regime adds value in
transitional/uncertain periods" hypothesis only if regime beats GARCH_t (QLIKE
mean lower) AND the DM test is significant (p<0.05) for the "uncertain" bucket
specifically; otherwise REJECT/INCONCLUSIVE. Not recomputed or moved after the
fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.eval.scores import QLIKE_PROXY_FLOOR, qlike
from model_lab.eval.tests_dm import diebold_mariano
from model_lab.regime.markov_switching import fit_and_filter_2state
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "regime_conditional_value"
# Buckets on dist_from_half = |high_vol_prob - 0.5|: SMALL dist means
# high_vol_prob is near 0.5 (the classifier is genuinely torn between the two
# regimes -- "uncertain"); LARGE dist means high_vol_prob is near 0 or 1 (the
# classifier is confident which regime we're in).
BUCKET_EDGES = {"uncertain": (0.0, 0.1), "transitional": (0.1, 0.4), "confident": (0.4, 0.5)}


def _qlike_loss_vec(y_true: np.ndarray, sigma2_hat: np.ndarray) -> np.ndarray:
    ratio = np.clip(y_true, QLIKE_PROXY_FLOOR, None) / np.clip(sigma2_hat, QLIKE_PROXY_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


@dataclass(frozen=True)
class BucketResult:
    bucket: str
    n: int
    qlike_regime: float
    qlike_garch_t: float
    regime_beats_garch_t: bool
    dm_stat: float
    dm_p: float


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}

    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        n = len(r)
        train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - 1})

        regime_res = fit_and_filter_2state(r, train_ends)
        garch_fc = gjr_garch_t_forecast(r, 1, train_ends).variance_forecast
        target = forward_realised_variance(r, 1)

        valid = [
            i
            for i in oos_indices
            if not np.isnan(regime_res.regime_variance_forecast[i]) and not np.isnan(garch_fc[i])
        ]
        if not valid:
            summary[ticker] = {"status": "SKIPPED", "reason": "no valid OOS points"}
            continue
        idx = np.array(valid)
        y = target[idx]
        dist_from_half = np.abs(regime_res.high_vol_prob[idx] - 0.5)

        buckets: dict[str, BucketResult] = {}
        for bucket_name, (lo, hi) in BUCKET_EDGES.items():
            mask = (dist_from_half >= lo) & (dist_from_half < hi) if hi < 0.5 else (dist_from_half >= lo) & (dist_from_half <= hi)
            n_bucket = int(mask.sum())
            if n_bucket < 30:
                buckets[bucket_name] = BucketResult(bucket_name, n_bucket, float("nan"), float("nan"), False, float("nan"), float("nan"))
                continue
            y_b = y[mask]
            regime_fc_b = regime_res.regime_variance_forecast[idx][mask]
            garch_fc_b = garch_fc[idx][mask]
            q_regime = qlike(y_b, regime_fc_b).value
            q_garch = qlike(y_b, garch_fc_b).value
            regime_loss = _qlike_loss_vec(y_b, regime_fc_b)
            garch_loss = _qlike_loss_vec(y_b, garch_fc_b)
            dm = diebold_mariano(regime_loss, garch_loss, h=1)
            buckets[bucket_name] = BucketResult(
                bucket_name, n_bucket, q_regime, q_garch, q_regime < q_garch, dm.dm_stat, dm.p_value
            )

        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps({b: asdict(v) for b, v in buckets.items()}, indent=2), encoding="utf-8")
        summary[ticker] = {b: asdict(v) for b, v in buckets.items()}
        print(
            f"{ticker}: "
            + " | ".join(
                f"{b}: n={v.n} regime={v.qlike_regime:.4f} garch={v.qlike_garch_t:.4f} "
                f"beats={v.regime_beats_garch_t} p={v.dm_p:.4f}"
                for b, v in buckets.items()
            )
        )

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
