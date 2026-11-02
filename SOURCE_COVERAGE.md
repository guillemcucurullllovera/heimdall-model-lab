# Source-to-destination coverage

144 first-party source modules, 123 test files, 30 runners/utilities and 14 schemas retained.

Every identified source implementation path in all four inspected branches is retained; alternate versions are explicitly superseded, not silently lost. The detailed machine-readable report includes every source entry, exact exclusion reason, source and destination SHA-256, tests and experiments. Local integration defaults are sanitized. No original scientific implementation is excluded for lack of private datasets.

## Sources

| Branch | Commit |
|---|---|
| `model/model2-challengers-20260930` | `c9bb47ae80e879dde6de4131f7e076594a24adec` |
| `research/model-lab-phase2-20261002` | `dda3e83a9320345e30c441edc8b028be2680760d` |
| `model2-remediation-review-4229338-20261002` | `c738825de1622d65d85429832e02931a4a3ce8d2` |
| `model/model1-foundation-20260930` | `a565f1ee00f2626a25c74745de54ea34b5b858dc` |

Phase-2 versions win shared paths. Challenger-only implementations, including economic evaluation, combined-volatility tails, prospective and evidence adapters, remain included.

## Included families

- EWMA and trailing realized variance
- GJR-GARCH-t
- EGARCH-t
- Mean and NaN-robust median forecast combinations
- GPD EVT, unconditional tails and GJR-filtered historical simulation
- Combined-volatility filtered GPD tail ablation
- Normal-inverse-gamma BOCPD and CUSUM
- Two-state Gaussian Markov switching
- Empirical/Ledoit-Wolf/OAS/robust covariance Mahalanobis anomalies
- Isolation Forest
- DM/HAC, SPA/MCS, multiple testing, conformal, e-processes, VaR/ES backtests
- Purged expanding/CPCV splits and leakage diagnostics
- Volatility targeting, risk-reduction utility and tail-capital efficiency

## Missing families

- No additional requested family missing from identified source; no neural, rough-volatility or multivariate GARCH implementation found

## Exclusions

Private result archives, operational handoffs and internal reports are excluded: their dataset rights and historical evidence are not established for redistribution. Source-era scientific commentary is retained with explicit historical-claim limitations. Private environment examples and proprietary-root metadata are replaced by release configuration. No third-party package source or dataset is vendored.

## Full module mapping

| Source module | Destination | Branch |
|---|---|---|
| `src/model_lab/__init__.py` | `src/model_lab/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anchor/__init__.py` | `src/model_lab/anchor/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anchor/builder.py` | `src/model_lab/anchor/builder.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anchor/contracts.py` | `src/model_lab/anchor/contracts.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anchor/storage.py` | `src/model_lab/anchor/storage.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anchor/verifier.py` | `src/model_lab/anchor/verifier.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anomaly/__init__.py` | `src/model_lab/anomaly/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anomaly/covariance_models.py` | `src/model_lab/anomaly/covariance_models.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anomaly/evaluate.py` | `src/model_lab/anomaly/evaluate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anomaly/isolation_forest_model.py` | `src/model_lab/anomaly/isolation_forest_model.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/anomaly/panel.py` | `src/model_lab/anomaly/panel.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/canon/__init__.py` | `src/model_lab/canon/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/canon/jcs.py` | `src/model_lab/canon/jcs.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/changepoint/__init__.py` | `src/model_lab/changepoint/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/changepoint/bocpd.py` | `src/model_lab/changepoint/bocpd.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/changepoint/cusum.py` | `src/model_lab/changepoint/cusum.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/changepoint/evaluate.py` | `src/model_lab/changepoint/evaluate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/command_center_export/__init__.py` | `src/model_lab/command_center_export/__init__.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/amc_promotion_evaluator.py` | `src/model_lab/command_center_export/amc_promotion_evaluator.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/builder.py` | `src/model_lab/command_center_export/builder.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/empirical_evaluator.py` | `src/model_lab/command_center_export/empirical_evaluator.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/gates_data.py` | `src/model_lab/command_center_export/gates_data.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/ledger_data.py` | `src/model_lab/command_center_export/ledger_data.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/model_registry_data.py` | `src/model_lab/command_center_export/model_registry_data.py` | `model/model2-challengers-20260930` |
| `src/model_lab/command_center_export/registry_builder.py` | `src/model_lab/command_center_export/registry_builder.py` | `model/model2-challengers-20260930` |
| `src/model_lab/compare/__init__.py` | `src/model_lab/compare/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/compare/vol_model_comparison.py` | `src/model_lab/compare/vol_model_comparison.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/__init__.py` | `src/model_lab/contracts/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/amc_promotion_evaluation.py` | `src/model_lab/contracts/amc_promotion_evaluation.py` | `model/model2-challengers-20260930` |
| `src/model_lab/contracts/base.py` | `src/model_lab/contracts/base.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/chart_spec.py` | `src/model_lab/contracts/chart_spec.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/dataset_snapshot.py` | `src/model_lab/contracts/dataset_snapshot.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/economic_evaluation.py` | `src/model_lab/contracts/economic_evaluation.py` | `model/model2-challengers-20260930` |
| `src/model_lab/contracts/evidence_report.py` | `src/model_lab/contracts/evidence_report.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/ledger_record.py` | `src/model_lab/contracts/ledger_record.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/licence.py` | `src/model_lab/contracts/licence.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/model2_export.py` | `src/model_lab/contracts/model2_export.py` | `model/model2-challengers-20260930` |
| `src/model_lab/contracts/model_lab_registry.py` | `src/model_lab/contracts/model_lab_registry.py` | `model/model2-challengers-20260930` |
| `src/model_lab/contracts/model_output.py` | `src/model_lab/contracts/model_output.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/model_version.py` | `src/model_lab/contracts/model_version.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/option_snapshot.py` | `src/model_lab/contracts/option_snapshot.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/prereg.py` | `src/model_lab/contracts/prereg.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/task.py` | `src/model_lab/contracts/task.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/contracts/transition_request.py` | `src/model_lab/contracts/transition_request.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/__init__.py` | `src/model_lab/data/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/credentials.py` | `src/model_lab/data/credentials.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/pit_macro.py` | `src/model_lab/data/pit_macro.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/pit_query.py` | `src/model_lab/data/pit_query.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/__init__.py` | `src/model_lab/data/providers/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/_http.py` | `src/model_lab/data/providers/_http.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/alpaca.py` | `src/model_lab/data/providers/alpaca.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/fred_alfred.py` | `src/model_lab/data/providers/fred_alfred.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/marketdata_app.py` | `src/model_lab/data/providers/marketdata_app.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/providers/massive.py` | `src/model_lab/data/providers/massive.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/registry.py` | `src/model_lab/data/registry.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/snapshot.py` | `src/model_lab/data/snapshot.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/data/snapshot_registry.py` | `src/model_lab/data/snapshot_registry.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/economic_evaluation/__init__.py` | `src/model_lab/economic_evaluation/__init__.py` | `model/model2-challengers-20260930` |
| `src/model_lab/economic_evaluation/diagnostics.py` | `src/model_lab/economic_evaluation/diagnostics.py` | `model/model2-challengers-20260930` |
| `src/model_lab/economic_evaluation/risk_reduction_utility.py` | `src/model_lab/economic_evaluation/risk_reduction_utility.py` | `model/model2-challengers-20260930` |
| `src/model_lab/economic_evaluation/tail_capital_efficiency.py` | `src/model_lab/economic_evaluation/tail_capital_efficiency.py` | `model/model2-challengers-20260930` |
| `src/model_lab/economic_evaluation/vol_targeting.py` | `src/model_lab/economic_evaluation/vol_targeting.py` | `model/model2-challengers-20260930` |
| `src/model_lab/eval/__init__.py` | `src/model_lab/eval/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/calibration.py` | `src/model_lab/eval/calibration.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/conformal.py` | `src/model_lab/eval/conformal.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/eprocess.py` | `src/model_lab/eval/eprocess.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/leakage.py` | `src/model_lab/eval/leakage.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/mcs_spa.py` | `src/model_lab/eval/mcs_spa.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/multiple_testing.py` | `src/model_lab/eval/multiple_testing.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/scores.py` | `src/model_lab/eval/scores.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/tests_dm.py` | `src/model_lab/eval/tests_dm.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/eval/var_es_backtests.py` | `src/model_lab/eval/var_es_backtests.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/__init__.py` | `src/model_lab/gate_v2/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/branch_f.py` | `src/model_lab/gate_v2/branch_f.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/branch_p.py` | `src/model_lab/gate_v2/branch_p.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/branch_s.py` | `src/model_lab/gate_v2/branch_s.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/cli.py` | `src/model_lab/gate_v2/cli.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/contracts.py` | `src/model_lab/gate_v2/contracts.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/formulas.py` | `src/model_lab/gate_v2/formulas.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/signing.py` | `src/model_lab/gate_v2/signing.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/v1_freeze.py` | `src/model_lab/gate_v2/v1_freeze.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/gate_v2/verdict.py` | `src/model_lab/gate_v2/verdict.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/governance/__init__.py` | `src/model_lab/governance/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/governance/contracts.py` | `src/model_lab/governance/contracts.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/governance/prereg_gate.py` | `src/model_lab/governance/prereg_gate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/governance/signer_registry.py` | `src/model_lab/governance/signer_registry.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/governance/verifier.py` | `src/model_lab/governance/verifier.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/__init__.py` | `src/model_lab/ledger/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/anchor.py` | `src/model_lab/ledger/anchor.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/chain.py` | `src/model_lab/ledger/chain.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/cli.py` | `src/model_lab/ledger/cli.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/signing.py` | `src/model_lab/ledger/signing.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/storage.py` | `src/model_lab/ledger/storage.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/verifier.py` | `src/model_lab/ledger/verifier.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/ledger/writer.py` | `src/model_lab/ledger/writer.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/options_archive/__init__.py` | `src/model_lab/options_archive/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/options_archive/anchor.py` | `src/model_lab/options_archive/anchor.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/options_archive/ingester.py` | `src/model_lab/options_archive/ingester.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/options_archive/storage.py` | `src/model_lab/options_archive/storage.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/options_archive/verifier.py` | `src/model_lab/options_archive/verifier.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/prospective/__init__.py` | `src/model_lab/prospective/__init__.py` | `model/model2-challengers-20260930` |
| `src/model_lab/prospective/capture.py` | `src/model_lab/prospective/capture.py` | `model/model2-challengers-20260930` |
| `src/model_lab/prospective/live_extend.py` | `src/model_lab/prospective/live_extend.py` | `model/model2-challengers-20260930` |
| `src/model_lab/prospective/live_fetch.py` | `src/model_lab/prospective/live_fetch.py` | `model/model2-challengers-20260930` |
| `src/model_lab/prospective/records.py` | `src/model_lab/prospective/records.py` | `model/model2-challengers-20260930` |
| `src/model_lab/prospective/resolve.py` | `src/model_lab/prospective/resolve.py` | `model/model2-challengers-20260930` |
| `src/model_lab/publish/__init__.py` | `src/model_lab/publish/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/publish/anchor.py` | `src/model_lab/publish/anchor.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/publish/builder.py` | `src/model_lab/publish/builder.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/publish/verify.py` | `src/model_lab/publish/verify.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/regime/__init__.py` | `src/model_lab/regime/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/regime/evaluate.py` | `src/model_lab/regime/evaluate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/regime/markov_switching.py` | `src/model_lab/regime/markov_switching.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/__init__.py` | `src/model_lab/replay/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/amc_root.py` | `src/model_lab/replay/amc_root.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/contracts.py` | `src/model_lab/replay/contracts.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/journal_reader.py` | `src/model_lab/replay/journal_reader.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/observer.py` | `src/model_lab/replay/observer.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/replay/parity.py` | `src/model_lab/replay/parity.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/research/__init__.py` | `src/model_lab/research/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/research/folds.py` | `src/model_lab/research/folds.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/shadow/__init__.py` | `src/model_lab/shadow/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/shadow/build.py` | `src/model_lab/shadow/build.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/shadow/build_combo.py` | `src/model_lab/shadow/build_combo.py` | `model/model2-challengers-20260930` |
| `src/model_lab/splitter/__init__.py` | `src/model_lab/splitter/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/splitter/cpcv.py` | `src/model_lab/splitter/cpcv.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/splitter/expanding.py` | `src/model_lab/splitter/expanding.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/splitter/purge_embargo.py` | `src/model_lab/splitter/purge_embargo.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tail/__init__.py` | `src/model_lab/tail/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tail/candidate_filtered_gpd_fhs.py` | `src/model_lab/tail/candidate_filtered_gpd_fhs.py` | `model/model2-challengers-20260930` |
| `src/model_lab/tail/conditional.py` | `src/model_lab/tail/conditional.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tail/evaluate.py` | `src/model_lab/tail/evaluate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tail/gpd.py` | `src/model_lab/tail/gpd.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tail/unconditional.py` | `src/model_lab/tail/unconditional.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tasks/__init__.py` | `src/model_lab/tasks/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/tasks/registry.py` | `src/model_lab/tasks/registry.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/__init__.py` | `src/model_lab/vol/__init__.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/baselines.py` | `src/model_lab/vol/baselines.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/candidate_spec.py` | `src/model_lab/vol/candidate_spec.py` | `model/model2-challengers-20260930` |
| `src/model_lab/vol/combination.py` | `src/model_lab/vol/combination.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/data.py` | `src/model_lab/vol/data.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/evaluate.py` | `src/model_lab/vol/evaluate.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/garch_models.py` | `src/model_lab/vol/garch_models.py` | `research/model-lab-phase2-20261002` |
| `src/model_lab/vol/targets.py` | `src/model_lab/vol/targets.py` | `research/model-lab-phase2-20261002` |
