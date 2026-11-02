"""Immutable publication directory builder — §18.

"immutable `evidence.json`, `charts/*.json`, `manifest.json` (SHA-256 of
every file + ledger head + model version hash), `model_card.json`,
`datasheet.json`; `data_authority = PROVISIONAL` only when hashes
recompute." MODEL-1 produces and tests these files; it changes nothing in
the Command Center itself.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.chart_spec import ChartSpec, DataAuthority, PublicationManifest
from model_lab.contracts.evidence_report import EvidenceReport

PUBLISH_ROOT = Path("./local-data/publish")


def publication_dir(model_id: str, model_version_hash: str, as_of: str) -> Path:
    return PUBLISH_ROOT / model_id / model_version_hash / as_of


def build_publication(
    *,
    model_id: str,
    model_version_hash: str,
    as_of: str,
    ledger_head_hash: str,
    evidence_report: EvidenceReport,
    charts: list[ChartSpec],
    model_card: dict,
    datasheet: dict,
) -> PublicationManifest:
    """Write an immutable publication directory and return its manifest
    (also written to `manifest.json`). Every write happens before the
    manifest is computed, so `manifest.file_hashes` is always accurate at
    write time — later tampering is what `verify_publication` (verify.py)
    detects, not this function."""
    out_dir = publication_dir(model_id, model_version_hash, as_of)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}

    def _write(rel_path: str, content: str) -> None:
        path = out_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        file_hashes[rel_path] = sha256_bytes(content.encode("utf-8"))

    _write("evidence.json", evidence_report.model_dump_json(indent=2))
    for chart in charts:
        _write(f"charts/{chart.chart_id}.json", chart.model_dump_json(indent=2))
    _write("model_card.json", json.dumps(model_card, indent=2, sort_keys=True))
    _write("datasheet.json", json.dumps(datasheet, indent=2, sort_keys=True))

    manifest = PublicationManifest(
        model_id=model_id,
        model_version_hash=model_version_hash,
        ledger_head_hash=ledger_head_hash,
        published_at_utc=datetime.now(UTC),
        file_hashes=file_hashes,
        data_authority=DataAuthority.PROVISIONAL,
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
