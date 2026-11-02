# Release validation

Local Python 3.12 validation, 2026-11-02:

- Complete offline suite: **548 passed, 85 skipped, zero failures/errors**.
- Ruff: all source, tests, scripts and experiments pass.
- Mypy: all 144 source files pass.
- Seeded synthetic experiment: 200 finite held-out outputs for every exercised
  family (EWMA, GJR, EGARCH, median ensemble, BOCPD, Markov, Mahalanobis,
  Isolation Forest, GPD-FHS VaR/ES); exact BOCPD prefix invariance passes.
- Wheel and source distribution build; no data estate, report archives, vendor
  package trees or Git history in either artifact.
- Isolated installed-wheel model/oracle tests: **161 passed, zero skips**; import
  confirmed from the isolated environment's `site-packages`, not the source tree.
  CI independently repeats this check on Linux.
- Dependency vulnerability audit: no known vulnerabilities in the locked environment.
- Gitleaks working-tree and complete proposed-public-history scans pass.

The 85 skips are explicit unavailable integrations: 65 exact private-estate
assertions listed in PRIVATE_ESTATE_TESTS.json, 19 pre-existing data/replay-dependent
skips, and one live-network test. No scientific algorithm test is hidden by an
expected-failure marker. Skipped tests are **not validated**.

The initial unscoped run had 548 passes, 56 failures, 10 setup errors and 19 skips.
PRIVATE_ESTATE_TESTS.json documents these failures without private traceback paths.
One live FRED test failed provider authentication; network access is now opt-in.
No private manifest, historical report or dataset is fabricated to turn a failed
historical assertion into a pass. With the original excluded inputs you can run:

```sh
uv run pytest --private-estate       # original estate assertions
uv run pytest --network -m network  # own provider credentials and terms required
```

Original numerical warnings (degenerate bootstrap data) and NumPy/calendar
deprecations remain visible. Synthetic checks establish execution and numerical
properties only, not historical empirical performance, market calibration or
investment usefulness. See LIMITATIONS.md for known estimator caveats.
