"""Model Lab Empirical Evaluation / Research Validation Layer.

Answers "how can this model's value be tested empirically, from its own
actual mathematical purpose" -- NOT "how do we force this model into a
P&L chart." Three evaluation shapes, chosen per model family by what the
model actually estimates, never uniformly applied:

1. **Volatility-targeting overlay backtest** (`vol_targeting.py`) -- the
   ONE legitimate, literature-grounded economic use of a variance
   forecast: scale a buy-and-hold position's exposure inversely to the
   forecast volatility (Moreira & Muir 2017, "Volatility-Managed
   Portfolios", Journal of Finance 72(4):1611-1644). This is a REAL,
   established technique, not invented for this task -- it requires no
   directional signal, no entry/exit timing, nothing a pure variance
   forecast does not actually produce. Applied to every T-VOL/T-STATE
   model that produces a variance forecast (naive, EWMA, GARCH_t,
   EGARCH_t, the MODEL-2 combination, and regime_2state).

2. **Tail capital-efficiency comparison** (`tail_capital_efficiency.py`)
   -- the appropriate decision-utility metric for a tail-risk model:
   for the SAME real coverage (already backtested via Kupiec/
   Christoffersen in the tail ladder), which model requires a SMALLER
   capital buffer (VaR/ES magnitude)? Reuses already-committed tail-
   ladder evidence; computes no new forecast.

3. **Risk-reduction decision-utility check** (`risk_reduction_utility.py`)
   -- the appropriate decision-utility metric for anomaly/changepoint
   detectors: if a risk manager cut exposure to zero on flagged days,
   what real forward-return impact would that have had? Not an alpha
   claim -- a risk-avoidance utility check, reusing already-committed
   scores and REAL realized returns.

Every assumption (target volatility, transaction-cost rate, leverage
cap) is a disclosed, pre-registered, round-number choice -- never tuned
after seeing results, never presented as a measured AMC-specific cost.
No model is forced into a strategy its own output cannot support.
"""
