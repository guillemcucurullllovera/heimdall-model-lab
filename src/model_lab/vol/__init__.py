"""MODEL-2A volatility forecast ladder (research/shadow only).

Not part of MODEL-1's frozen evidence foundation. Implements the classical
volatility forecast baselines the real, acquired six-ETF universe genuinely
supports today (naive trailing RV, EWMA, GJR-GARCH-t, EGARCH-t) against the
forward realised-variance target T-VOL already declares
(`tasks/registry.py`: `variable="forward_realised_variance"`,
`aggregation="sum_of_squared_returns_or_range_proxy"`).

T-VOL itself remains BLOCKED (the `implied_vol_regressors` requirement is
still genuinely unmet -- confirmed live this session, see
`reports/MODEL2A_DATA_VOL_TAIL_BASELINES.md`), so nothing here produces
promotion evidence or feeds AMC/PAPER_AUTOPILOT in any way. Every output is
shadow research evidence only.
"""
