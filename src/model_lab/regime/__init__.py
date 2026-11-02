"""MODEL-2D: regime/state modelling (research/shadow only). Implements a
2-state Gaussian Markov-switching mean+variance model (Hamilton 1989-style,
`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`) as a
first, bounded pass at T-STATE's frozen family. TVTP, jump models, and
finite HSMM are explicitly NOT implemented this phase (a scope decision --
see `reports/MODEL2B_2C_2D_COVARIANCE_CHANGEPOINT_REGIME.md`); Student-t
(rather than Gaussian) regime innovations are also not supported by
`statsmodels`' implementation and are noted as a known approximation, not
silently glossed over.

T-STATE/T-DOWNSTREAM remain `BLOCKED` on `market_state_features`
(`tasks/registry.py`) -- this package runs on the real six-ETF return
series instead, disclosed explicitly, never claimed as satisfying those
tasks."""
