#!/usr/bin/env python3
"""Ingest every non-empty AMC candidate file into the option archive.

Read-only against the AMC root; writes only to
./local-data/options_archive/ and .../raw/.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.canon.jcs import canonical_sha256
from model_lab.contracts.option_snapshot import OptionArchiveChainManifest
from model_lab.options_archive import (
    ARCHIVE_PATH,
    OptionArchiveFile,
    finalize_and_append,
    ingest_candidate_file,
)
from model_lab.options_archive.verifier import verify_archive_chain
from model_lab.replay.amc_root import CYCLES_DIR

MANIFEST_DIR = Path("./local-data/options_archive/manifests")


def _experiment_id_for(cycle_id: str) -> str:
    date_suffix = cycle_id.rsplit("-", 1)[-1]
    if date_suffix == "20260908":
        return "amc4-prospective-v1-20260908"
    if date_suffix == "20260925":
        return "amc4-prospective-v2-20260925"
    raise ValueError(f"unrecognised cycle_id date suffix: {cycle_id}")


def main() -> None:
    archive = OptionArchiveFile(ARCHIVE_PATH)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    all_candidate_files = sorted(CYCLES_DIR.glob("cycle-*_candidates.json"))
    summary = []
    for path in all_candidate_files:
        cycle_id = path.name.removesuffix("_candidates.json")
        content = json.loads(path.read_text(encoding="utf-8"))
        n_candidates = len(content)
        if n_candidates == 0:
            summary.append({"cycle_id": cycle_id, "n_candidates": 0, "n_legs_ingested": 0, "status": "EMPTY_SKIPPED"})
            continue

        experiment_id = _experiment_id_for(cycle_id)
        drafts = ingest_candidate_file(path, amc_experiment_id=experiment_id, amc_cycle_id=cycle_id)
        finalized = finalize_and_append(archive, drafts)

        manifest = OptionArchiveChainManifest(
            manifest_id=f"manifest-{cycle_id}",
            n_contracts=len(finalized),
            coverage_vs_expected_chain="unknown (candidate subset, not a full listed chain)",
            calendar_holiday_flags=[],
            ingest_code_hash=canonical_sha256({"ingester": "options_archive.ingester", "version": 1}),
            source_licence_snapshot_hash=canonical_sha256({"provider": "alpaca", "status": "UNRESOLVED"}),
            archive_head_hash=finalized[-1].record_hash,
        )
        (MANIFEST_DIR / f"{manifest.manifest_id}.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        summary.append(
            {
                "cycle_id": cycle_id,
                "n_candidates": n_candidates,
                "n_legs_ingested": len(finalized),
                "status": "INGESTED",
            }
        )
        print(f"ingested {cycle_id}: {n_candidates} candidates -> {len(finalized)} leg-records")

    records = archive.read_all()
    result = verify_archive_chain(records)
    print(f"\narchive total records: {len(records)}; chain verifies: {result.ok}")
    if not result.ok:
        for v in result.violations[:10]:
            print(f"  VIOLATION {v}")

    out = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "archive_path": str(ARCHIVE_PATH),
        "total_records": len(records),
        "chain_verifies": result.ok,
        "per_cycle": summary,
    }
    (MANIFEST_DIR / "ingest_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {MANIFEST_DIR / 'ingest_summary.json'}")


if __name__ == "__main__":
    main()
