# Heimdall Model Lab

Complete first-party classical-model research system: volatility and tail forecasts,
forecast ensembles, online changepoints, regime switching, anomaly detection,
statistical inference and leakage-aware validation. This parent repository is
separate from the narrower `model-lab-forecast-evaluation` package.

## Reproduce

```sh
uv sync --locked --extra dev --extra splitter-oracle --python 3.12
uv run --no-sync pytest -q
uv run --no-sync python experiments/synthetic.py
uv build
```

Python 3.12. Do not install this alongside `model-lab-forecast-evaluation` in the
same environment: both intentionally provide the `model_lab` namespace.

## Scientific architecture

| Layer | Modules |
|---|---|
| Forecasts | `vol`: trailing RV, EWMA, GJR-GARCH-t, EGARCH-t, mean/median ensembles |
| Risk | `tail`: GPD EVT, unconditional and GJR/combined-volatility filtered historical simulation |
| State | `changepoint`: causal NIG BOCPD, CUSUM; `regime`: two-state Hamilton filtering |
| Anomalies | `anomaly`: empirical, Ledoit-Wolf, OAS, robust Mahalanobis and Isolation Forest |
| Inference | `eval`: proper scores, DM/HAC, SPA/MCS, multiplicity, conformal, e-processes, VaR/ES |
| Validation | `splitter`, `research`: purging, embargo, CPCV, expanding folds; leakage diagnostics |
| Economic research | `economic_evaluation`: volatility targeting, risk utility, tail-capital efficiency |
| Evidence | typed contracts, point-in-time datasets, licensing firewall, ledger, governance, publication |
| Runners | `scripts/run_*`: ladders, comparisons, ablations, shadow/prospective research |

See [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md),
[SOURCE_COVERAGE.md](SOURCE_COVERAGE.md), [LIMITATIONS.md](LIMITATIONS.md) and
[PROVENANCE.md](PROVENANCE.md). Original scientific implementations and tests are
retained, not replaced by stubs. Historical source assertions are not release results.

## Data and integrations

No dataset, provider credential, fitted production parameter archive or private
freeze/signing manifest is distributed. Defaults point to `./local-data`, which is
ignored by Git. Dataset-backed runners fail closed without correctly licensed,
registered snapshots. Provider clients require your own credentials and terms of
use; they do not grant redistribution rights. Replay inputs are local and read-only.

Apache-2.0 applies to authorized first-party source. Original implementation:
Francisco. Release engineering: Guillem Cucurull Llovera. See NOTICE.
