"""Genuine prospective-evidence collection for the MODEL-2G candidate
(Phase 5 of the post-MODEL-2G non-review continuation).

Critical distinction this package exists to enforce: a historical
walk-forward replay (everything in `vol/evaluate.py`, `reports/vol_ladder/`,
etc.) scores a forecast against an outcome that was ALREADY PRESENT in the
frozen research dataset before the forecast was ever computed -- it
validates the pipeline's mechanics, but it is not prospective evidence.
Genuine prospective evidence requires an outcome that did not exist, or was
not yet knowable, in any of this project's data stores at the moment the
forecast was produced and recorded.

How this package gets that: `vol/data.load_return_series` reads ONLY the
frozen, already-registered `DatasetSnapshot` (see its own docstring: "No
new data is fetched here"), whose last date is fixed. This package's
`capture` step freezes an "information cutoff" at that same last date and
produces a forecast for the NEXT (unknown) trading day. Its `resolve` step,
run later (possibly much later, possibly immediately if real time has
already moved past the target date), makes a REAL, READ-ONLY, live API call
(`live_fetch.py`, via the already-authorised `MassiveClient` -- same
provider, same read-only discipline MODEL-2A's own VIX-availability checks
used) to ask whether the real world has actually produced a bar for that
date yet. If not, the observation stays PENDING -- never fabricated, never
silently skipped. If so, the realized outcome is appended as a SEPARATE,
immutable `ResolvedObservation` record that references the original
`PendingObservation`'s own content hash; the pending record itself is never
mutated.

Explicitly NOT done here, matching every other Model Lab shadow/research
package in this repository:
- no scheduler/cron/service is created or activated (this package is
  invoked manually via `scripts/run_prospective_capture.py`, exactly once
  per invocation, same discipline as `run_shadow_snapshot.py`);
- no write to the real hash-chained evidence ledger (`ledger/`) -- records
  live under `reports/prospective/`, a clearly separate, disclosed area;
- no write to the frozen `DatasetSnapshot`/data-estate store -- the live
  fetch in `live_fetch.py` is held entirely in memory and in this
  package's own `reports/prospective/` output, never merged back into
  `data/snapshots/` or any registry the rest of Model Lab reads from;
- no AMC/PAPER_AUTOPILOT/Command Center/Macro/Quantum read or write;
- no trading, no order submission, no position of any kind;
- `ModelOutput.decision_relevant` is always False on every record this
  package produces.
"""
