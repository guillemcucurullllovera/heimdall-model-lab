"""Seeded model smoke experiment. No historical performance claim."""
import json

import numpy as np

from model_lab.anomaly.covariance_models import mahalanobis_d2_series
from model_lab.anomaly.isolation_forest_model import isolation_forest_score_series
from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma
from model_lab.regime.markov_switching import fit_and_filter_2state
from model_lab.tail.conditional import gjr_t_filtered_gpd_fhs_forecast
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast


def run():
    rng = np.random.default_rng(20261102)
    returns = rng.standard_t(7, 700) * np.r_[np.full(350, .008), np.full(350, .02)]
    folds = [500]
    ewma = ewma_forecast(returns, 1)
    gjr = gjr_garch_t_forecast(returns, 1, folds).variance_forecast
    egarch = egarch_t_forecast(returns, 1, folds).variance_forecast
    combo = median_variance_combination(ewma, gjr, egarch)
    cp = bocpd_normal_inverse_gamma(returns)
    prefix = bocpd_normal_inverse_gamma(returns[:300])
    np.testing.assert_allclose(cp.changepoint_prob[:300], prefix.changepoint_prob, rtol=0, atol=0)
    regimes = fit_and_filter_2state(returns, folds)
    panel = np.column_stack([returns, rng.normal(0, .01, len(returns))])
    covariance = mahalanobis_d2_series(panel, folds, estimator='ledoit_wolf')
    forest = isolation_forest_score_series(panel, folds, n_estimators=40)
    tail = gjr_t_filtered_gpd_fhs_forecast(returns, folds, tail_levels=(.05,))
    outputs = {'ewma': ewma, 'gjr': gjr, 'egarch': egarch, 'median': combo,
               'bocpd': cp.changepoint_prob, 'markov': regimes.high_vol_prob,
               'mahalanobis': covariance.mahalanobis_d2, 'isolation_forest': forest.anomaly_score,
               'gpd_fhs_var': tail.var_forecast[.05], 'gpd_fhs_es': tail.es_forecast[.05]}
    counts = {k: int(np.isfinite(v[500:]).sum()) for k, v in outputs.items()}
    assert all(count > 0 for count in counts.values()), counts
    assert np.all(tail.es_forecast[.05][500:] >= tail.var_forecast[.05][500:])
    assert np.all((cp.changepoint_prob >= 0) & (cp.changepoint_prob <= 1))
    return {'data': 'seeded synthetic Student-t variance shift; NOT historical results',
            'seed': 20261102, 'finite_test_outputs': counts, 'bocpd_prefix_invariance': True}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
