#!/usr/bin/env python3
"""Phase 8 (MODEL-2 non-review continuation): is the candidate's own
dispersion across its 3 constituents (EWMA_0.94/GARCH_t/EGARCH_t) -- not a
native output of the candidate itself, but a cheap diagnostic computed
from exactly the same information the candidate already uses -- predictive
of when the candidate's OWN forecast error will be large? If so, this is a
genuine implementation-level (Layer C) architecture value: an abstention/
low-confidence signal, distinct from "a better forecast."

Pre-registered decision rule (stated before computing results): for the
3 eligible tickers (TLT/IEF/GLD, h=1), bucket OOS origins into tertiles of
NORMALIZED dispersion (max-min spread of the 3 constituents' forecasts,
divided by their median, computed CAUSALLY from information available at
that origin -- no look-ahead). RETAIN the "dispersion predicts error"
hypothesis only if the candidate's mean QLIKE loss in the TOP dispersion
tertile is higher than in the BOTTOM tertile, for a majority (>=2/3) of
the three tickers, AND the pooled (all 3 tickers) top-vs-bottom difference
is significant at DM p<0.05. Otherwise REJECT/INCONCLUSIVE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.eval.scores import QLIKE_PROXY_FLOOR
from model_lab.eval.tests_dm import diebold_mariano
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "confidence_abstention_probe"
ELIGIBLE_TICKERS = ("TLT", "IEF", "GLD")
HORIZON = 1


def _qlike_loss_vec(y_true: np.ndarray, sigma2_hat: np.ndarray) -> np.ndarray:
    ratio = np.clip(y_true, QLIKE_PROXY_FLOOR, None) / np.clip(sigma2_hat, QLIKE_PROXY_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    pooled_top_losses: list[float] = []
    pooled_bottom_losses: list[float] = []
    tickers_where_top_worse = 0

    for ticker in ELIGIBLE_TICKERS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        n = len(r)
        train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=HORIZON, min_train_groups=2)
        oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - HORIZON})

        ewma_fc = ewma_forecast(r, HORIZON, lam=0.94)
        garch_fc = gjr_garch_t_forecast(r, HORIZON, train_ends).variance_forecast
        egarch_fc = egarch_t_forecast(r, HORIZON, train_ends).variance_forecast
        target = forward_realised_variance(r, HORIZON)

        valid = [i for i in oos_indices if not (np.isnan(ewma_fc[i]) or np.isnan(garch_fc[i]) or np.isnan(egarch_fc[i]))]
        idx = np.array(valid)
        stacked = np.stack([ewma_fc[idx], garch_fc[idx], egarch_fc[idx]], axis=0)
        combo_fc = median_variance_combination(ewma_fc[idx], garch_fc[idx], egarch_fc[idx])
        dispersion = (stacked.max(axis=0) - stacked.min(axis=0)) / np.median(stacked, axis=0)

        y = target[idx]
        loss = _qlike_loss_vec(y, combo_fc)

        q1, q2 = np.percentile(dispersion, [100 / 3, 200 / 3])
        bottom_mask = dispersion <= q1
        top_mask = dispersion >= q2

        bottom_loss_mean = float(loss[bottom_mask].mean())
        top_loss_mean = float(loss[top_mask].mean())
        top_worse = top_loss_mean > bottom_loss_mean
        if top_worse:
            tickers_where_top_worse += 1
        pooled_top_losses.extend(loss[top_mask].tolist())
        pooled_bottom_losses.extend(loss[bottom_mask].tolist())

        summary[ticker] = {
            "n_oos": len(valid),
            "bottom_tertile_mean_loss": bottom_loss_mean,
            "top_tertile_mean_loss": top_loss_mean,
            "top_worse_than_bottom": top_worse,
            "dispersion_q1": float(q1),
            "dispersion_q2": float(q2),
        }
        print(f"{ticker}: n_oos={len(valid)} bottom_loss={bottom_loss_mean:.4f} top_loss={top_loss_mean:.4f} top_worse={top_worse}")

    n = min(len(pooled_top_losses), len(pooled_bottom_losses))
    dm = diebold_mariano(np.array(pooled_top_losses[:n]), np.array(pooled_bottom_losses[:n]), h=1)
    decision = "RETAIN" if tickers_where_top_worse >= 2 and dm.p_value < 0.05 and dm.dm_stat > 0 else "REJECT"
    summary["_decision"] = {
        "pre_registered_rule": "top dispersion tertile has higher mean QLIKE loss for >=2/3 tickers AND pooled DM p<0.05",
        "tickers_where_top_worse": tickers_where_top_worse,
        "pooled_dm_stat": dm.dm_stat,
        "pooled_dm_p": dm.p_value,
        "decision": decision,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\npooled DM (top vs bottom tertile loss): stat={dm.dm_stat:.4f} p={dm.p_value:.4f}")
    print(f"Decision: {decision} ({tickers_where_top_worse}/3 tickers show top-dispersion tertile worse)")
    print(f"wrote {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
