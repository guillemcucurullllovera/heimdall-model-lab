# Limitations and non-reproduced claims

Only tests and experiments explicitly reported in TEST_RESULTS.md are release
validation. Synthetic data verifies mechanics, not historical empirical performance.

Private six-ETF price snapshots, source registries, prospective captures, fitted
parameters, signing authorities and frozen evidence manifests are absent. Historical
ladder QLIKE rankings, DM significance, candidate eligibility, downstream economic
benefits, promotion readiness and shadow/prospective results cannot be reproduced
from this repository alone. Source-era commentary and the legacy candidate cell
configuration retain provenance; they are not independently endorsed results.

EGARCH multi-step simulation uses the source's seeded normal simulation draws even
though the fitted innovation distribution is Student-t. Thus it is not an exact
Student-t predictive simulation. Plausibility exclusions are guards, not a numerical
stability proof; common-sample comparisons may select away difficult origins.
Mean/median combination has no learned weights or native predictive interval.

Some estimators enforce a minimum training length with `max(boundary-purge, minimum)`;
at short boundaries this can reduce the intended purge. Use sufficiently long
training histories; do not assume arbitrary fold inputs satisfy leakage guarantees.
Tail fits are unstable with few exceedances and ES may not exist for fitted shape
at or above one. Markov optimizer convergence and identification are separate;
original convergence metadata requires caution. Source-era BOCPD run-length MAP
indices after pruning are not guaranteed to equal the actual elapsed run length.

No hardware execution, production integration, investment performance, neural
forecasting, rough volatility or multivariate GARCH claim is made. Evidence
infrastructure is research software, not a production security attestation.
