"""InputBundle/1 and DecisionDiff/1 — MODEL1_foundation_contracts.md §16.

```
InputBundle/1 { cycle_id, journal_prefix_hash, candidate_file_hashes[],
  source_health_snapshot_hash, observed_at_utc, root_path, experiment_id }
```

"a true pre-decision `input_bundle_hash` emitted by the AMC itself would
require an AMC change (out of scope, needs authorisation)." This
InputBundle is built observer-side, after the fact, from whatever the AMC
already persisted — never a substitute for that.

DecisionDiff/1 is referenced (`reports/amc_integration_and_value_attribution.md
§4`, not read this session) but not given a verbatim field list in the three
contracts docs MODEL-1 was built from. It is designed here to satisfy what
§16 explicitly requires: "replay parity = rerunning the incumbent on a
recorded bundle reproduces the recorded decision fingerprint for 100% of
cycles that have one." Because MODEL-1 does not reimplement the AMC-3
deterministic solver (§16: "a separate audit item," out of scope here),
`replayed_decision_fingerprint` and `match` are honestly `None` /
NOT_COMPUTABLE whenever no independent reimplementation exists — never
fabricated as a false match.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex


class InputBundle(ModelLabBaseModel):
    schema_name: str = "InputBundle"
    schema_version: int = 1

    cycle_id: str
    journal_prefix_hash: Sha256Hex
    journal_prefix_byte_length: int = Field(ge=0)
    candidate_file_hashes: list[Sha256Hex] = Field(default_factory=list)
    source_health_snapshot_hash: Sha256Hex | None = None
    observed_at_utc: datetime
    root_path: str
    experiment_id: str


class ParityStatus(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"


class DecisionDiff(ModelLabBaseModel):
    schema_name: str = "DecisionDiff"
    schema_version: int = 1

    cycle_id: str
    experiment_id: str
    model_version_hash: Sha256Hex | None = Field(
        default=None, description="None when comparing against the AMC's own recorded incumbent, not a challenger"
    )
    recorded_decision_fingerprint: str | None = None
    replayed_decision_fingerprint: str | None = Field(
        default=None,
        description="None whenever no independent selection reimplementation exists for this cycle "
        "(honest NOT_COMPUTABLE, never a fabricated match)",
    )
    parity_status: ParityStatus
    notes: str
    observed_at_utc: datetime
