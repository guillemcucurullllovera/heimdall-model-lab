"""ModelVersion/1 — MODEL1_foundation_contracts.md §11.

```
ModelVersion/1 { model_id, model_version_hash = H("ml/model/v1" || experiment_fingerprint ||
  parameter_artifact_hashes), family, spec_hash, code_tree_hash,
  parameter_artifacts[{path, sha256}] (JSON/npz -- NO pickle), data_snapshot_ids[], task_ids[],
  lifecycle_state, influence_tier, tier_ceiling_signed, predecessor_hash, refit_lane: FAST|SLOW,
  limit_table_hash, model_card_hash }
```

Lifecycle: RESEARCH -> REPRODUCED -> WALK_FORWARD -> SHADOW -> PROSPECTIVE_CHALLENGER ->
PROMOTION_ELIGIBLE -> PROMOTED(T1|T2|T3), with side states REJECTED / SUPERSEDED / DEGRADED /
QUARANTINED. "Registry reads are pure functions of the ledger (replaying the ledger reproduces
the current state)."
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.task import InfluenceTier


class LifecycleState(StrEnum):
    RESEARCH = "RESEARCH"
    REPRODUCED = "REPRODUCED"
    WALK_FORWARD = "WALK_FORWARD"
    SHADOW = "SHADOW"
    PROSPECTIVE_CHALLENGER = "PROSPECTIVE_CHALLENGER"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
    PROMOTED = "PROMOTED"
    # side states
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    DEGRADED = "DEGRADED"
    QUARANTINED = "QUARANTINED"


# The forward "happy path" lineage, in order. Any state may also transition
# to a side state (REJECTED/SUPERSEDED/DEGRADED/QUARANTINED); side states
# other than SUPERSEDED are terminal for influence purposes (a QUARANTINED
# or DEGRADED model can only move to SUPERSEDED by a brand-new ModelVersion,
# never resume forward from where it was).
FORWARD_LIFECYCLE_ORDER: list[LifecycleState] = [
    LifecycleState.RESEARCH,
    LifecycleState.REPRODUCED,
    LifecycleState.WALK_FORWARD,
    LifecycleState.SHADOW,
    LifecycleState.PROSPECTIVE_CHALLENGER,
    LifecycleState.PROMOTION_ELIGIBLE,
    LifecycleState.PROMOTED,
]

SIDE_STATES: frozenset[LifecycleState] = frozenset(
    {
        LifecycleState.REJECTED,
        LifecycleState.SUPERSEDED,
        LifecycleState.DEGRADED,
        LifecycleState.QUARANTINED,
    }
)


def is_valid_lifecycle_transition(before: LifecycleState, after: LifecycleState) -> bool:
    """A transition is valid iff it is exactly one step forward on the happy
    path, or a move into any side state from any non-terminal state. This is
    the *structural* rule; the influence-tier invariant (validator+operator
    for any influence increase) is enforced separately in governance."""
    if after in SIDE_STATES:
        return before not in (LifecycleState.PROMOTED,) or after == LifecycleState.SUPERSEDED
    try:
        bi = FORWARD_LIFECYCLE_ORDER.index(before)
        ai = FORWARD_LIFECYCLE_ORDER.index(after)
    except ValueError:
        return False
    return ai == bi + 1


class RefitLane(StrEnum):
    FAST = "FAST"
    SLOW = "SLOW"


class ParameterArtifact(ModelLabBaseModel):
    path: str
    sha256: Sha256Hex

    @field_validator("path")
    @classmethod
    def _no_pickle(cls, v: str) -> str:
        if v.endswith((".pkl", ".pickle", ".pt", ".joblib")):
            raise ValueError(f"parameter artifact must not be a pickle-family format: {v!r}")
        if not v.endswith((".json", ".npz", ".npy", ".ndjson")):
            raise ValueError(
                f"parameter artifact must be JSON/npz/npy/ndjson, got {v!r} "
                "(§11: 'NO pickle')"
            )
        return v


def compute_model_version_hash(
    experiment_fingerprint: str, parameter_artifact_hashes: list[str]
) -> str:
    """model_version_hash = H('ml/model/v1' || experiment_fingerprint || parameter_artifact_hashes)"""
    payload = "ml/model/v1" + "||" + experiment_fingerprint + "||" + "|".join(
        sorted(parameter_artifact_hashes)
    )
    return sha256_bytes(payload.encode("utf-8"))


class ModelVersion(ModelLabBaseModel):
    schema_name: str = "ModelVersion"
    schema_version: int = 1

    model_id: str
    model_version_hash: Sha256Hex
    family: str
    spec_hash: Sha256Hex
    code_tree_hash: Sha256Hex
    parameter_artifacts: list[ParameterArtifact]
    data_snapshot_ids: list[str]
    task_ids: list[str]
    lifecycle_state: LifecycleState
    influence_tier: InfluenceTier
    tier_ceiling_signed: bool = Field(
        description="True only if a signed TransitionRequest authorised this tier"
    )
    predecessor_hash: Sha256Hex | None = None
    refit_lane: RefitLane
    limit_table_hash: Sha256Hex | None = None
    model_card_hash: Sha256Hex | None = None

    @field_validator("model_version_hash")
    @classmethod
    def _hash_format(cls, v: str) -> str:
        return v
