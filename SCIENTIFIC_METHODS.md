# Scientific methods and reading map

The original module docstrings preserve detailed mathematical conventions and
implementation rationale; this guide separates those methods from private results.

- **Volatility:** RiskMetrics EWMA and trailing realized variance predict cumulative
  future squared log returns. GJR-GARCH(1,1) includes leverage and Student-t fitted
  innovations; EGARCH(1,1)-t models log variance. Fits are periodically refreshed
  using purged training prefixes. `arch` supplies estimation and fixed-parameter
  forecasting. Forecasts are aligned to their end-of-day origin, never their target.
- **Combination:** Bates–Granger equal weighting and Stock–Watson-style median
  combination. The median is NaN-robust; downstream shared-sample evaluation can
  still require every constituent to be available. Report exclusions explicitly.
- **Tails:** peaks-over-threshold generalized Pareto fits, historical VaR/ES and
  conditional GPD filtered historical simulation. GJR standardized residuals supply
  the fitted tail; next-period conditional scale transforms residual quantiles.
  The combination-scaled version is an ablation, not a validated trading strategy.
- **Changepoints:** Adams–MacKay-style BOCPD uses NIG conjugate updates and Student-t
  predictive densities; this implementation uses a fresh-prior reset likelihood
  (a reset-before-observation convention). Causal expanding prior scale avoids
  whole-series variance leakage. Run-length pruning is approximate.
- **Regimes:** Hamilton Gaussian two-state switching mean/variance via statsmodels.
  Regimes are variance-ordered, filtered rather than smoothed, and propagated one
  transition ahead. Degenerate low-persistence fits are excluded.
- **Anomalies:** Mahalanobis squared distance with empirical, Ledoit–Wolf, OAS and
  minimum-covariance-determinant estimates; Liu et al. Isolation Forest. Training
  uses prior purged folds; scores are contemporaneous anomaly diagnostics.
- **Inference:** Diebold–Mariano with HAC correction; stationary-bootstrap SPA/MCS;
  multiple-testing correction; coverage/calibration, conformal intervals,
  sequential e-processes and VaR/ES backtests. Statistical significance is not
  economic usefulness or an unconditional guarantee under arbitrary dependence.
- **Validation:** label-interval purging and embargo, expanding walk-forward and
  combinatorial purged cross-validation. The contiguous-prefix shortcut applies
  only to expanding folds, never general CPCV training sets.

References: RiskMetrics Technical Document (1996); Glosten, Jagannathan & Runkle
(1993); Nelson (1991); Bates & Granger (1969); Stock & Watson (2004); McNeil & Frey
(2000); Adams & MacKay (2007), arXiv:0710.3742; Hamilton (1989); Ledoit & Wolf
(2004); Liu, Ting & Zhou (2008); Diebold & Mariano (1995); Hansen (2005); Hansen,
Lunde & Nason (2011); López de Prado, *Advances in Financial Machine Learning*
(2018). These references identify methods, not copied third-party source.
