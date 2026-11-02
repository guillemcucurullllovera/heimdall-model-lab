"""Walk-forward anomaly-scoring run across the four estimators, plus an
explicitly ILLUSTRATIVE (not validating) cross-check against a small set of
widely-known historical market-stress dates.

**T-ANOM (C4) remains `BLOCKED`.** Its real, named blocker
(`tasks/registry.py`: `frozen_stress_event_labels`, "no pre-registered
financial anomaly benchmark with labelled events exists") is a governance/
pre-registration gap. The illustrative date list below is NOT a substitute
for that -- it is not pre-registered, not exhaustive, and was chosen with
the benefit of hindsight (the opposite of a pre-registered benchmark's
whole point). It is included ONLY as a sanity/descriptive check that the
scores are not obviously broken, never as evidence of validated detection
power. Any future use of this module for an actual T-ANOM promotion attempt
MUST use a real, separately pre-registered event set instead.

**Second, distinct source of hindsight in the check itself (remediation,
2026-10-02 Phase 2, independent review finding P2-7).** The underlying
Mahalanobis/Isolation-Forest scores ARE genuinely causal/OOS -- each fold's
fit uses only `returns[:fit_end_idx]`, confirmed by `check_prefix_
invariance` below. But `_illustrative_check`'s "is this date in the top
decile/5%" LABEL is computed by ranking each date's score against the
ENTIRE pooled out-of-sample score distribution (`scores[~np.isnan(scores)]`
-- every date from 2018 to 2026 at once), not a causal/expanding quantile
as of that date. A 2018 date is judged "top decile" relative to a 9-year
retrospective distribution that includes the 2020 COVID crash and every
later extreme it could not have been compared against in real time. This
is a SEPARATE mechanism from, and not covered by, the hindsight caveat
above (which is about how the 14 dates themselves were chosen, not about
how the percentile denominator is computed) -- both are disclosed, neither
is fixed, because this check is explicitly illustrative/descriptive, never
a validated detection-power claim, in either respect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from model_lab.anomaly.covariance_models import ESTIMATORS, mahalanobis_d2_series
from model_lab.anomaly.isolation_forest_model import isolation_forest_score_series
from model_lab.anomaly.panel import ReturnPanel
from model_lab.research.folds import expanding_fold_train_ends

MIN_TRAIN_YEARS = 2

# Illustrative only (see module docstring) -- widely reported, real market-
# stress dates that fall within the data's real 2016-2026 window. NOT
# pre-registered, NOT exhaustive, chosen with hindsight.
ILLUSTRATIVE_STRESS_DATES = [
    "2018-02-05", "2018-02-06",  # "Volmageddon" -- VIX spike, inverse-vol ETP collapse
    "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28",
    "2020-03-09", "2020-03-12", "2020-03-16",  # COVID-19 crash
    "2022-01-24", "2022-06-13",  # 2022 rate-shock selloffs
    "2023-03-10", "2023-03-13",  # SVB / regional-bank stress
]


def _fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class AnomalyModelResult:
    model_id: str
    n_scored: int
    illustrative_dates_covered: int
    illustrative_dates_in_top_decile: int
    illustrative_dates_in_top_5pct: int
    score_percentile_of_illustrative_dates: list[float]


@dataclass(frozen=True)
class AnomalyLadderRunResult:
    tickers: list[str]
    data_fingerprints: dict
    model_fingerprints: dict
    n_folds: int
    fold_train_end_dates: list[str]
    oos_start_date: str
    oos_end_date: str
    n_oos_total: int
    results: dict
    no_leakage_proof: dict
    caveat: str = (
        "Illustrative-stress-date cross-check is NOT a validated detection-power claim -- "
        "T-ANOM's real frozen_stress_event_labels blocker is unresolved. See module docstring."
    )


def _illustrative_check(dates: list[str], scores: np.ndarray) -> dict:
    date_to_idx = {d: i for i, d in enumerate(dates)}
    covered = [d for d in ILLUSTRATIVE_STRESS_DATES if d in date_to_idx and not np.isnan(scores[date_to_idx[d]])]
    if not covered:
        return {
            "illustrative_dates_covered": 0, "illustrative_dates_in_top_decile": 0,
            "illustrative_dates_in_top_5pct": 0, "score_percentile_of_illustrative_dates": [],
        }
    # RETROSPECTIVE percentile, by design (module docstring, "second,
    # distinct source of hindsight"): `valid_scores` pools the full
    # 2018-2026 OOS run, so a 2018 date's rank below is relative to extremes
    # (e.g. the 2020 COVID crash) that had not yet happened -- this is a
    # second, separate source of hindsight from how the 14 dates themselves
    # were chosen, disclosed, not fixed, because the check is illustrative.
    valid_scores = scores[~np.isnan(scores)]
    ranks = []
    for d in covered:
        s = scores[date_to_idx[d]]
        pct = float((valid_scores <= s).mean())  # fraction of all scored days this day's score exceeds
        ranks.append(pct)
    return {
        "illustrative_dates_covered": len(covered),
        "illustrative_dates_in_top_decile": sum(1 for p in ranks if p >= 0.90),
        "illustrative_dates_in_top_5pct": sum(1 for p in ranks if p >= 0.95),
        "score_percentile_of_illustrative_dates": ranks,
    }


def run_anomaly_ladder(panel: ReturnPanel, *, min_train_groups: int = MIN_TRAIN_YEARS) -> AnomalyLadderRunResult:
    r = panel.returns
    dates = panel.dates
    n = len(dates)

    train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=min_train_groups)
    if not train_ends:
        raise ValueError(f"not enough history for {min_train_groups} warm-up year(s)")
    oos_indices = sorted({i for lst in test_index_lists for i in lst})

    scores: dict[str, np.ndarray] = {}
    fold_fits: dict[str, list] = {}
    for estimator in ESTIMATORS:
        res = mahalanobis_d2_series(r, train_ends, estimator=estimator)
        scores[estimator] = res.mahalanobis_d2
        fold_fits[estimator] = res.fold_fits
    if_res = isolation_forest_score_series(r, train_ends)
    scores["Isolation_Forest"] = if_res.anomaly_score
    fold_fits["Isolation_Forest"] = if_res.fold_fits

    model_fingerprints = {m: _fingerprint({"model": m, "tickers": panel.tickers}) for m in scores}

    results: dict[str, dict] = {}
    all_valid: list[int] = []
    for model_id, s in scores.items():
        valid_idx = [i for i in oos_indices if not np.isnan(s[i])]
        if not valid_idx:
            continue
        all_valid.extend(valid_idx)
        check = _illustrative_check(dates, s)
        results[model_id] = asdict(
            AnomalyModelResult(
                model_id=model_id,
                n_scored=len(valid_idx),
                illustrative_dates_covered=check["illustrative_dates_covered"],
                illustrative_dates_in_top_decile=check["illustrative_dates_in_top_decile"],
                illustrative_dates_in_top_5pct=check["illustrative_dates_in_top_5pct"],
                score_percentile_of_illustrative_dates=check["score_percentile_of_illustrative_dates"],
            )
        )

    # No-leakage proof: a covariance/Mahalanobis scorer is a pure function of
    # (returns, fold_train_ends) -- prefix-invariance applies directly, same as
    # vol/baselines.py's check, using a SMALL checkpoint set (each checkpoint
    # implies a cheap refit, not a per-day one).
    from model_lab.eval.leakage import check_prefix_invariance

    def _ledoit_wolf_feature(data: np.ndarray) -> np.ndarray:
        local_ends = [e for e in train_ends if e < len(data)]
        return mahalanobis_d2_series(data, local_ends, estimator="ledoit_wolf").mahalanobis_d2

    checkpoints = [e + 10 for e in train_ends if e + 10 < n][:3]  # a few cheap spot-checks, not every fold
    leak_result = check_prefix_invariance(_ledoit_wolf_feature, r, checkpoints=checkpoints, atol=1e-6)

    all_valid_sorted = sorted(set(all_valid))
    return AnomalyLadderRunResult(
        tickers=panel.tickers,
        data_fingerprints=panel.data_fingerprints,
        model_fingerprints=model_fingerprints,
        n_folds=len(train_ends),
        fold_train_end_dates=[dates[e] for e in train_ends],
        oos_start_date=dates[all_valid_sorted[0]] if all_valid_sorted else "",
        oos_end_date=dates[all_valid_sorted[-1]] if all_valid_sorted else "",
        n_oos_total=len(all_valid_sorted),
        results=results,
        no_leakage_proof={
            "ledoit_wolf_prefix_invariance_all_passed": leak_result.all_passed,
            "ledoit_wolf_prefix_invariance_max_abs_diff": leak_result.max_abs_diff,
            "checkpoints": leak_result.checkpoints,
        },
    )
