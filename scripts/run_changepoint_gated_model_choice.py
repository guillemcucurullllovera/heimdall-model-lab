#!/usr/bin/env python3
"""Layer C research probe (implementation edge, not a new forecast model):
even though MODEL-2C found BOCPD is a much weaker STANDALONE detector than a
cross-sectional anomaly scorer, and MODEL-2A found GARCH_t beats EWMA_0.94
on QLIKE pooled across all days for 4/6 tickers -- does BOCPD's
changepoint_prob have value as a GATING signal, i.e. does EWMA_0.94's faster
adaptation specifically help on days BOCPD flags as elevated-changepoint-
probability, even on tickers/horizons where GARCH_t wins on average?

This tests "regime/changepoint info gating model selection" (Layer C) rather
than "regime/changepoint info as a standalone forecast" (already answered,
negatively, by MODEL-2D).

Pre-registered decision rule (declared BEFORE computing results, exactly the
discipline instructed -- not moved after seeing the numbers): "elevated"
means `changepoint_prob > 5 * hazard` (hazard=1/250 is BOCPD's own disclosed
parameter throughout this repo, so 5x is a fixed, motivated, non-cherry-picked
multiple of it, not a threshold reverse-engineered from this experiment's own
results) evaluated causally at each t (BOCPD is online; no future data is
used to decide whether t is "elevated"). RETAIN the "changepoint-gated
switching helps" hypothesis only if, on the "elevated" subset specifically,
EWMA_0.94's QLIKE is lower than GARCH_t's AND the Diebold-Mariano test is
significant at p<0.05; otherwise REJECT/INCONCLUSIVE. No other threshold
was tried and discarded."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.eval.scores import QLIKE_PROXY_FLOOR, qlike
from model_lab.eval.tests_dm import diebold_mariano
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "changepoint_gated_model_choice"
HAZARD = 1.0 / 250.0
ELEVATED_THRESHOLD = 5.0 * HAZARD


def _qlike_loss_vec(y_true: np.ndarray, sigma2_hat: np.ndarray) -> np.ndarray:
    ratio = np.clip(y_true, QLIKE_PROXY_FLOOR, None) / np.clip(sigma2_hat, QLIKE_PROXY_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


@dataclass(frozen=True)
class SubsetResult:
    subset: str
    n: int
    qlike_ewma: float
    qlike_garch_t: float
    ewma_beats_garch_t: bool
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

        bocpd_res = bocpd_normal_inverse_gamma(r, hazard=HAZARD)
        ewma_fc = ewma_forecast(r, 1, lam=0.94)
        garch_fc = gjr_garch_t_forecast(r, 1, train_ends).variance_forecast
        target = forward_realised_variance(r, 1)

        valid = [i for i in oos_indices if not np.isnan(ewma_fc[i]) and not np.isnan(garch_fc[i])]
        idx = np.array(valid)
        y = target[idx]
        elevated_mask = bocpd_res.changepoint_prob[idx] > ELEVATED_THRESHOLD

        results: dict[str, SubsetResult] = {}
        for subset_name, mask in [("elevated", elevated_mask), ("normal", ~elevated_mask)]:
            n_sub = int(mask.sum())
            if n_sub < 30:
                results[subset_name] = SubsetResult(subset_name, n_sub, float("nan"), float("nan"), False, float("nan"), float("nan"))
                continue
            y_s = y[mask]
            ewma_s = ewma_fc[idx][mask]
            garch_s = garch_fc[idx][mask]
            q_ewma = qlike(y_s, ewma_s).value
            q_garch = qlike(y_s, garch_s).value
            dm = diebold_mariano(_qlike_loss_vec(y_s, ewma_s), _qlike_loss_vec(y_s, garch_s), h=1)
            results[subset_name] = SubsetResult(subset_name, n_sub, q_ewma, q_garch, q_ewma < q_garch, dm.dm_stat, dm.p_value)

        out_path = OUT_DIR / f"{ticker}.json"
        out_path.write_text(json.dumps({k: asdict(v) for k, v in results.items()}, indent=2), encoding="utf-8")
        summary[ticker] = {k: asdict(v) for k, v in results.items()}
        print(
            f"{ticker}: "
            + " | ".join(
                f"{k}: n={v.n} ewma={v.qlike_ewma:.4f} garch={v.qlike_garch_t:.4f} "
                f"ewma_beats={v.ewma_beats_garch_t} p={v.dm_p:.4f}"
                for k, v in results.items()
            )
        )

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
