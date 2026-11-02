"""Hash-recompute guard — §18: "`data_authority = PROVISIONAL` only when
hashes recompute." A publication directory is read back and every file
listed in `manifest.json` (except the manifest itself) is re-hashed; any
mismatch or missing file makes that file's chart(s) render `UNAVAILABLE`
rather than silently serving stale/tampered content, per the ChartSpec
contract's own `unavailable`/`value` mutual-exclusion rule (never a 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.chart_spec import DataAuthority, PublicationManifest


@dataclass(frozen=True)
class PublicationVerificationResult:
    ok: bool
    data_authority: DataAuthority
    mismatched_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)


def verify_publication(publication_dir: Path) -> PublicationVerificationResult:
    manifest_path = publication_dir / "manifest.json"
    manifest = PublicationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    mismatched: list[str] = []
    missing: list[str] = []
    for rel_path, expected_hash in manifest.file_hashes.items():
        path = publication_dir / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        actual_hash = sha256_bytes(path.read_bytes())
        if actual_hash != expected_hash:
            mismatched.append(rel_path)

    ok = not mismatched and not missing
    return PublicationVerificationResult(
        ok=ok,
        data_authority=DataAuthority.PROVISIONAL if ok else DataAuthority.UNAVAILABLE,
        mismatched_files=mismatched,
        missing_files=missing,
    )
