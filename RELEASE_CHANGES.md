# Changes from the source branches

- Union of challenger-only scientific/evidence additions with the latest phase-2
  shared implementations, including purge, multiplicity and robust-ensemble fixes.
  Original private clones and published standalone packages are unchanged.
- Private production filesystem roots replaced with `./local-data`; the credential
  file default is user-owned `~/.config/model-lab/data.env`. No credential values ship.
- Original proprietary package metadata replaced under the recorded owner-authorized
  Apache-2.0 first-party license grant. Original implementation attribution retained.
- SPA boolean renamed to `benchmark_not_rejected`, carrying forward the already
  public standalone correction: non-rejection is not proof of benchmark dominance.
  Corresponding internal consumers and comparison-output field names updated;
  bootstrap statistics and decision comparisons are numerically unchanged.
- Obsolete foundation-only package description corrected to include actual models.
- Actual model dependencies made part of the core install; development/oracle extras
  and Python 3.12 locked. No third-party source is vendored.
- Added seeded multi-family synthetic experiment, two independent ensemble tests,
  complete source coverage verification, offline CI and installed-wheel tests.
- Original private-data assertion tests remain unmodified and are opt-in by exact
  node ID in PRIVATE_ESTATE_TESTS.json; network-marked tests are also opt-in. This
  does not turn their original failures into passes or validate their historical claims.

No historical results, missing datasets, model placeholders, author dates or
contribution history are invented. Known numerical/statistical limitations remain
explicit in LIMITATIONS.md rather than silently claiming a complete scientific audit.
