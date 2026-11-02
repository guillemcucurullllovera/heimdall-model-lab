"""MODEL-2 Command-Center export: a read-only, hash-verifiable projection
of the already-committed MODEL-2 research/shadow/prospective evidence into
`contracts/model2_export.py`'s `Model2CommandCenterExport` contract.

Scope and boundaries (identical to every other research/shadow package in
this repository):
- read-only with respect to the Model Lab's own evidence (`reports/`,
  `vol/candidate_spec.py`, shadow/prospective records) -- this package
  never recomputes or re-derives a result, it only projects already-
  committed JSON into one consumable document;
- NEVER writes to, reads from, or references AMC/PAPER_AUTOPILOT/Command
  Center/Macro/Quantum code or state;
- NEVER sets `amc_promotion_authorized=True` (enforced by a test, not just
  a convention -- see `tests/unit/command_center_export/`);
- NEVER advances a `LifecycleState` or promotion-gate state beyond what
  the underlying evidence has actually established;
- produces output under `reports/command_center_export/` only -- never
  writes into any Command Center repository or service (this repository
  has no Command Center code to write into; exporting the data for a
  SEPARATE Command Center repository to later consume is the entire
  point, and is explicitly out of this repository's own scope to build).
"""
