"""MODEL-2F: shadow/prospective evaluation infrastructure (research/
shadow only). Builds REAL, validating MODEL-1 `ModelVersion`
(`lifecycle_state=SHADOW`, `tier_ceiling_signed=False`) and `ModelOutput`
(`decision_relevant=False`) records for the vol/tail ladder's own
challenger models -- not a new ad hoc schema, the actual frozen contracts
MODEL-1 committed for exactly this purpose (`model_output.py`'s own
docstring: "nothing in this repository consumes ModelOutput records" --
this package is the first thing that PRODUCES one, still nothing consumes
it).

Explicitly NOT done here, per instruction:
- no scheduler/cron/service is created or activated;
- no signed TransitionRequest, no tier_ceiling_signed=True;
- no write to the real hash-chained evidence ledger (`ledger/`) -- shadow
  records are written to `reports/shadow/`, a clearly separate, disclosed
  area, not the ledger's own storage;
- no AMC/PAPER_AUTOPILOT/Command Center/Macro/Quantum read or write.

Running `scripts/run_shadow_snapshot.py` ONCE manually produces one dated
snapshot demonstrating the capability -- it is not scheduled to repeat.
"""
