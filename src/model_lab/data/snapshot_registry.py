"""DatasetSnapshotRegistry — P0-1 remediation (adversarial review finding
C.1): "Licence/promotion firewall is bypassable — source authority is not
cryptographically or structurally bound, only trusted metadata... A
fabricated DataRequirement can claim PROMOTION_ELIGIBLE and produce
PROMOTION_READY without resolving a real DatasetSnapshot."

This is the "authoritative registry/data snapshot layer" the remediation
brief requires (P0-1 items 1/3): an immutable-for-the-life-of-the-process
lookup from `dataset_id` (== `DatasetSnapshot.snapshot_id`) to the real,
on-disk snapshot's authoritative fields. `contracts/task.py`'s `Task`
validator (not this module) is what actually ENFORCES that a
`DataRequirement` claiming a `dataset_id`/`research_state` matches what this
registry says — this module only answers "what does the real data say,"
never itself decides what's allowed.

Built by scanning `heimdall-model-lab-data/snapshots/*/manifest.json` — each
manifest IS a `DatasetSnapshot.model_dump_json()` (see `data/snapshot.py`).
No network access, no mutation of the data estate.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.dataset_snapshot import DatasetSnapshot, PitRow
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.data.snapshot import SNAPSHOTS_ROOT


class ResolvedAuthority(ModelLabBaseModel):
    """What a `DataRequirement.dataset_id` claim resolves to, per P0-1
    requirement 4 ("Bind at minimum: dataset_id, snapshot/content hash,
    licence_status, research_state, relevant adjustment state,
    provenance/source registry identity")."""

    dataset_id: str
    content_hash: Sha256Hex = Field(description="DatasetSnapshot.logical_content_hash")
    licence_status: LicenceStatus
    research_state: ResearchState
    adjustment_kind: str = Field(description="DatasetSnapshot.adjustments.kind")
    source_id: str = Field(description="DatasetSnapshot.source_id -- provenance/source registry identity")


def resolved_authority_from_snapshot(snapshot: DatasetSnapshot) -> ResolvedAuthority:
    return ResolvedAuthority(
        dataset_id=snapshot.snapshot_id,
        content_hash=snapshot.logical_content_hash,
        licence_status=LicenceStatus(snapshot.licence_status),
        research_state=ResearchState(snapshot.research_state),
        adjustment_kind=snapshot.adjustments.kind,
        source_id=snapshot.source_id,
    )


class DatasetSnapshotRegistry:
    """Immutable (post-construction) lookup. `DatasetSnapshot`'s own
    contract validator already guarantees `research_state` is mechanically
    consistent with `licence_status` for every entry here (see
    `contracts/dataset_snapshot.py::_check_research_state_matches_licence`)
    — this registry does not re-derive that, it only exposes it, which is
    exactly why resolving THROUGH a real `DatasetSnapshot` (rather than
    trusting a `DataRequirement`'s own fields) closes the P0-1 gap: the
    snapshot's own cross-field validator is the thing a fabricated
    `DataRequirement` could never have been forced to satisfy.
    """

    def __init__(self, snapshots: dict[str, DatasetSnapshot]) -> None:
        self._snapshots = dict(snapshots)

    def resolve(self, dataset_id: str) -> ResolvedAuthority | None:
        snap = self._snapshots.get(dataset_id)
        return resolved_authority_from_snapshot(snap) if snap is not None else None

    def get_snapshot(self, dataset_id: str) -> DatasetSnapshot | None:
        """The full real DatasetSnapshot, for callers that need structural
        fields beyond `ResolvedAuthority` (e.g. `pit_class`, for P1-3's
        mechanical macro-PIT bundling) -- still always the real, registered
        object, never a caller-supplied stand-in."""
        return self._snapshots.get(dataset_id)

    def get_rows(self, dataset_id: str) -> list[PitRow] | None:
        """Full row content for a dataset -- loads from disk if the
        snapshot stores `rows_path` rather than embedding `rows` directly.

        MODEL-2A P1 remediation (`MODEL1_REMEDIATION_REREVIEW.md` §I,
        "new finding... P0-4's companion-dataset check validates existence,
        not content"): `Task`'s validator (`contracts/task.py`) uses this to
        actually read a companion corporate-actions snapshot's rows, rather
        than treating `registry.resolve(...)` returning non-None as
        sufficient evidence of a zero-event claim.

        Returns `None` only when `dataset_id` itself doesn't resolve to any
        registered snapshot -- a real, on-disk snapshot with genuinely zero
        rows returns `[]`, a meaningfully different (and valid) answer from
        "unknown"."""
        snap = self._snapshots.get(dataset_id)
        if snap is None:
            return None
        if snap.rows is not None:
            return snap.rows
        assert snap.rows_path is not None  # DatasetSnapshot's own validator guarantees exactly one is set
        from model_lab.data.snapshot import load_rows_ndjson

        return load_rows_ndjson(snap.rows_path)

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    @classmethod
    def from_snapshots(cls, snapshots: list[DatasetSnapshot]) -> DatasetSnapshotRegistry:
        return cls({s.snapshot_id: s for s in snapshots})

    @classmethod
    def empty(cls) -> DatasetSnapshotRegistry:
        """An explicitly empty registry -- every dataset_id resolution
        fails, so any DataRequirement claiming a dataset_id/research_state
        against it fails closed. NOT the same as "no registry at all"
        (which Task's validator refuses outright, per P0-1 requirement 5) --
        this is for tasks/specs that are deliberately not yet claiming any
        real dataset (dataset_id=None everywhere)."""
        return cls({})

    @classmethod
    def load_from_directory(cls, snapshots_root: Path = SNAPSHOTS_ROOT) -> DatasetSnapshotRegistry:
        """Scan `<snapshots_root>/*/manifest.json` (read-only) and build the
        registry from whatever real DatasetSnapshots are actually on disk."""
        snapshots: dict[str, DatasetSnapshot] = {}
        if not snapshots_root.exists():
            return cls(snapshots)
        for manifest_path in sorted(snapshots_root.glob("*/manifest.json")):
            snap = DatasetSnapshot.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            snapshots[snap.snapshot_id] = snap
        return cls(snapshots)
