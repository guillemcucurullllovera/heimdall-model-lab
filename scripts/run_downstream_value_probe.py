#!/usr/bin/env python3
"""Phase 7 (MODEL-2 non-review continuation): downstream economic/risk
value probe, isolated from AMC. Does the MODEL-2G candidate's vol-
forecasting edge (TLT/IEF/GLD at h=1, per `vol/candidate_spec.py`) produce
better VaR/ES coverage when it scales T-TAIL's existing Filtered
Historical Simulation tail shape, vs the incumbent GARCH_t scaling?

Pre-registered decision rule (stated before computing results): for
TLT/IEF/GLD specifically (the candidate's own demonstrated h=1 edge),
RETAIN "the vol-forecast edge has downstream risk value" only if the
Combo-scaled FHS's Christoffersen LR_cc p-value at the 95% level (the
level MODEL-2A's own incumbent already fails for TLT/IEF/GLD) is HIGHER
(less rejecting) than GARCH_t-scaled FHS's own, for a majority (>=2/3) of
these three tickers. For SPY/QQQ/USO (not eligible cells), no directional
claim is pre-registered either way -- reported for completeness/honesty,
not cherry-picked out."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.eval.var_es_backtests import (
    acerbi_szekely_es_statistic,
    christoffersen_lr_cc,
    hit_sequence,
    kupiec_lr_uc,
)
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tail.candidate_filtered_gpd_fhs import combo_filtered_gpd_fhs_forecast
from model_lab.tail.conditional import gjr_t_filtered_gpd_fhs_forecast
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "downstream_value_probe"
TAIL_LEVELS = (0.05, 0.025, 0.01)
ELIGIBLE_TICKERS = ("TLT", "IEF", "GLD")


@dataclass(frozen=True)
class LevelResult:
    p: float
    n_oos: int
    n_violations: int
    observed_rate: float
    kupiec_lr_uc_p: float
    christoffersen_lr_cc_p: float
    acerbi_szekely_z1: float


def _score(model, r: np.ndarray, oos_indices: list[int]) -> dict[str, dict]:
    losses = -r
    per_level: dict[str, dict] = {}
    for p in TAIL_LEVELS:
        valid_idx = [i for i in oos_indices if not np.isnan(model.var_forecast[p][i]) and not np.isnan(model.es_forecast[p][i])]
        if not valid_idx:
            continue
        idx_arr = np.array(valid_idx)
        target_arr = idx_arr + 1
        loss_oos = losses[target_arr]
        var_oos = model.var_forecast[p][idx_arr]
        es_oos = model.es_forecast[p][idx_arr]
        hits = hit_sequence(loss_oos, var_oos)
        uc = kupiec_lr_uc(hits, p)
        cc = christoffersen_lr_cc(hits, p)
        asz = acerbi_szekely_es_statistic(loss_oos, var_oos, es_oos, p)
        per_level[str(p)] = asdict(
            LevelResult(
                p=p, n_oos=len(valid_idx), n_violations=uc.n_violations, observed_rate=uc.observed_rate,
                kupiec_lr_uc_p=uc.p_value, christoffersen_lr_cc_p=cc.p_value, acerbi_szekely_z1=asz.z_stat,
            )
        )
    return per_level


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    summary: dict[str, dict] = {}
    eligible_better_at_95 = 0

    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        n = len(r)
        train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        seam_origins = {te - 1 for te in train_ends}
        oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - 1 and i not in seam_origins})

        incumbent = gjr_t_filtered_gpd_fhs_forecast(r, train_ends, tail_levels=TAIL_LEVELS)
        candidate = combo_filtered_gpd_fhs_forecast(r, train_ends, tail_levels=TAIL_LEVELS)

        incumbent_scores = _score(incumbent, r, oos_indices)
        candidate_scores = _score(candidate, r, oos_indices)
        summary[ticker] = {"incumbent_GARCH_t_scaled": incumbent_scores, "candidate_Combo_median_scaled": candidate_scores}

        p95 = str(0.05)
        inc_p = incumbent_scores[p95]["christoffersen_lr_cc_p"]
        cand_p = candidate_scores[p95]["christoffersen_lr_cc_p"]
        better = cand_p > inc_p
        if ticker in ELIGIBLE_TICKERS and better:
            eligible_better_at_95 += 1
        print(
            f"{ticker}: 95% Christoffersen LR_cc p -- incumbent(GARCH_t)={inc_p:.4f} "
            f"candidate(Combo_median)={cand_p:.4f} candidate_better={better} (eligible_ticker={ticker in ELIGIBLE_TICKERS})"
        )

    decision = "RETAIN" if eligible_better_at_95 >= 2 else "REJECT"
    summary["_decision"] = {
        "pre_registered_rule": "candidate better at 95% Christoffersen LR_cc for >=2/3 of TLT/IEF/GLD",
        "eligible_tickers_where_candidate_better": eligible_better_at_95,
        "decision": decision,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nDecision: {decision} ({eligible_better_at_95}/3 eligible tickers improved at the 95% level)")
    print(f"wrote {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
