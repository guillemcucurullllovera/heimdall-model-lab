"""Walk-forward VaR/ES backtesting for the tail ladder, using MODEL-1's
existing `eval.var_es_backtests` (Kupiec LR_uc, Christoffersen LR_cc,
Acerbi-Szekely) and `eval.scores.fz0_joint_var_es`.

Sign conventions (kept explicit throughout, the exact class of error the
USO corporate-action remediation exists to guard against elsewhere in this
codebase): `eval.var_es_backtests` uses POSITIVE loss magnitudes
(loss = -return, VaR/ES reported as positive numbers); `fz0_joint_var_es`
uses the SIGNED return convention (VaR/ES reported as negative numbers, the
alpha-quantile of returns). This module converts between them explicitly at
each call site rather than silently picking one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from model_lab.eval.scores import fz0_joint_var_es
from model_lab.eval.var_es_backtests import (
    acerbi_szekely_es_statistic,
    christoffersen_lr_cc,
    hit_sequence,
    kupiec_lr_uc,
)
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tail.conditional import gjr_t_filtered_gpd_fhs_forecast
from model_lab.tail.unconditional import unconditional_gpd_pot_forecast
from model_lab.vol.data import ReturnSeries

MIN_TRAIN_YEARS = 2
TAIL_LEVELS = (0.05, 0.025, 0.01)


def _fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class TailLevelResult:
    p: float
    n_oos: int
    n_violations: int
    observed_rate: float
    nominal_rate: float
    kupiec_lr_uc_stat: float
    kupiec_lr_uc_p: float
    christoffersen_lr_cc_stat: float
    christoffersen_lr_cc_p: float
    acerbi_szekely_z1: float
    fz0_mean: float


@dataclass(frozen=True)
class TailLadderRunResult:
    ticker: str
    dataset_id: str
    data_fingerprint: str
    model_fingerprints: dict
    n_folds: int
    fold_train_end_dates: list[str]
    oos_start_date: str
    oos_end_date: str
    n_oos_total: int
    results: dict  # model_id -> {p_str: TailLevelResult-as-dict}


def run_tail_ladder(series: ReturnSeries, *, min_train_groups: int = MIN_TRAIN_YEARS, tail_levels: tuple[float, ...] = TAIL_LEVELS) -> TailLadderRunResult:
    r = series.log_returns
    dates = series.dates[1:]
    n = len(r)
    losses = -r

    train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=min_train_groups)
    if not train_ends:
        raise ValueError(f"{series.ticker}: not enough history for {min_train_groups} warm-up year(s)")
    # A forecast stored at index i (both tail models, matching vol/garch_models'
    # origin convention) predicts the loss realised on r[i+1], NOT r[i] --
    # r[i] was already known at the moment the forecast was made. i == n-1 has
    # no r[i+1] and must be excluded.
    #
    # Remediation (independent adversarial review, finding P3-9): VaR/ES are
    # FLAT within each calendar-year fold's own assignment window
    # `[train_end_idx, next_boundary)`. Because scoring uses `target = i+1`,
    # the LAST origin of fold k (`next_boundary-1`) scores its target against
    # the FIRST day of fold k+1's test window (`next_boundary`) -- a real,
    # if tiny, cross-fold misattribution: fold k's (older, purged-training)
    # VaR estimate gets blamed for a day that's really in fold k+1's nominal
    # test year. Quantified directly: exactly one such seam origin per fold
    # boundary (8 live seams across the 9 folds used throughout this session
    # -- the 9th, at the very first fold boundary, was already naturally
    # excluded by the pre-existing NaN filter below, since no fold's
    # var_forecast covers anything before the first fold starts). Fixed by
    # excluding these exact seam origins from scoring entirely (neither
    # fold "wins" the ambiguous day) rather than silently misattributing
    # them -- out of ~2200 OOS points per ticker, this removes at most 8.
    seam_origins = {te - 1 for te in train_ends}
    oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - 1 and i not in seam_origins})

    uncond = unconditional_gpd_pot_forecast(r, train_ends, tail_levels=tail_levels)
    cond = gjr_t_filtered_gpd_fhs_forecast(r, train_ends, tail_levels=tail_levels)

    model_fingerprints = {
        uncond.model_id: _fingerprint(uncond.config),
        cond.model_id: _fingerprint(cond.config),
    }

    results: dict[str, dict] = {}
    for model in (uncond, cond):
        per_level: dict[str, dict] = {}
        for p in tail_levels:
            valid_idx = [i for i in oos_indices if not np.isnan(model.var_forecast[p][i]) and not np.isnan(model.es_forecast[p][i])]
            if not valid_idx:
                continue
            idx_arr = np.array(valid_idx)
            target_arr = idx_arr + 1  # the day each forecast at idx_arr actually covers
            loss_oos = losses[target_arr]
            var_oos = model.var_forecast[p][idx_arr]
            es_oos = model.es_forecast[p][idx_arr]

            hits = hit_sequence(loss_oos, var_oos)
            uc = kupiec_lr_uc(hits, p)
            cc = christoffersen_lr_cc(hits, p)
            asz = acerbi_szekely_es_statistic(loss_oos, var_oos, es_oos, p)

            # fz0 convention: signed returns, VaR/ES as NEGATIVE numbers.
            y_signed = r[target_arr]
            var_signed = -var_oos
            es_signed = -es_oos
            fz0 = fz0_joint_var_es(y_signed, var_signed, es_signed, p)

            per_level[str(p)] = asdict(
                TailLevelResult(
                    p=p,
                    n_oos=len(valid_idx),
                    n_violations=uc.n_violations,
                    observed_rate=uc.observed_rate,
                    nominal_rate=p,
                    kupiec_lr_uc_stat=uc.lr_stat,
                    kupiec_lr_uc_p=uc.p_value,
                    christoffersen_lr_cc_stat=cc.lr_stat,
                    christoffersen_lr_cc_p=cc.p_value,
                    acerbi_szekely_z1=asz.z_stat,
                    fz0_mean=fz0.value,
                )
            )
        results[model.model_id] = per_level

    any_level = tail_levels[0]
    all_valid = sorted({i for i in oos_indices if not np.isnan(uncond.var_forecast[any_level][i]) and not np.isnan(cond.var_forecast[any_level][i])})
    all_valid_targets = [i + 1 for i in all_valid]  # dates actually covered by the backtest, not the forecast origins

    return TailLadderRunResult(
        ticker=series.ticker,
        dataset_id=series.dataset_id,
        data_fingerprint=series.data_fingerprint,
        model_fingerprints=model_fingerprints,
        n_folds=len(train_ends),
        fold_train_end_dates=[dates[e] for e in train_ends],
        oos_start_date=dates[all_valid_targets[0]] if all_valid_targets else "",
        oos_end_date=dates[all_valid_targets[-1]] if all_valid_targets else "",
        n_oos_total=len(all_valid),
        results=results,
    )
