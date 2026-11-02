"""MODEL-2C: changepoint detection (research/shadow only). T-CHANGE's
frozen `naive="CUSUM"` and one of its two named `incumbents` ("BOCPD") are
implemented here; the second incumbent, Fearnhead's exact/particle-filter
multiple-changepoint method, is explicitly NOT implemented this phase (a
scope decision, not an oversight -- see
`reports/MODEL2B_2C_2D_COVARIANCE_CHANGEPOINT_REGIME.md`).

T-CHANGE itself remains `BLOCKED` on `market_state_features`
(`tasks/registry.py`) -- this package is applied to the real six-ETF return
series instead of T-CHANGE's frozen `SPX_proxy` universe, disclosed
explicitly, never claimed as satisfying that task."""
