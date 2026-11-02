#!/usr/bin/env python3
"""Export every MODEL-1 contract's JSON Schema to schemas/*.json (committed).

Run after any contract change: `uv run python scripts/export_json_schema.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.contracts import (
    ChartSpec,
    DatasetSnapshot,
    EvidenceReport,
    LedgerRecord,
    ModelOutput,
    ModelVersion,
    OptionSnapshotRecord,
    Prereg,
    PublicationManifest,
    RunManifest,
    Task,
    TransitionRequest,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"

MODELS = {
    "LedgerRecord.v1": LedgerRecord,
    "DatasetSnapshot.v1": DatasetSnapshot,
    "Task.v1": Task,
    "Prereg.v1": Prereg,
    "RunManifest.v1": RunManifest,
    "ModelVersion.v1": ModelVersion,
    "EvidenceReport.v1": EvidenceReport,
    "TransitionRequest.v1": TransitionRequest,
    "OptionSnapshotRecord.v1": OptionSnapshotRecord,
    "ChartSpec.v1": ChartSpec,
    "PublicationManifest.v1": PublicationManifest,
    "ModelOutput.v1": ModelOutput,
}


def main() -> None:
    SCHEMAS_DIR.mkdir(exist_ok=True)
    for name, model in MODELS.items():
        schema = model.model_json_schema()
        out = SCHEMAS_DIR / f"{name}.json"
        out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(SCHEMAS_DIR.parent)}")


if __name__ == "__main__":
    main()
