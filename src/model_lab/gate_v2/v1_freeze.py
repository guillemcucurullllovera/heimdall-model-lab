"""v1 freeze verification — MODEL1_foundation_contracts.md §15 / AT-10.

"v1 untouched -- the 33-file freeze manifest is signed by the operator and
a hash mismatch is logged as a 'v1 freeze violation'." This module reads
`gate_v1_freeze_manifest.json` (read-only) and, for each of the 33 `files[]`
entries, recomputes sha256 of the file at its `path` (also read-only --
under heimdall-v2, never written to) and compares to the manifest's stored
`sha256`.

Per the verified research extraction: "there is no single aggregate
'manifest hash' field that hashes all 33 files together... freeze integrity
is per-file (33 independent sha256 checks)." This module ALSO computes
`canonical_sha256(manifest_json)` as a separate, MODEL-1-side convenience
hash for identifying which exact manifest version was checked -- this is
NOT the same thing as an aggregate freeze hash and is documented as such.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from model_lab.canon.jcs import canonical_sha256

DEFAULT_MANIFEST_PATH = Path(
    "./local-data/manifests/gate_v1_freeze_manifest.json"
)


@dataclass(frozen=True)
class FileCheckResult:
    path: str
    manifest_sha256: str
    recomputed_sha256: str | None
    match: bool
    error: str | None = None


@dataclass(frozen=True)
class V1FreezeVerificationResult:
    manifest_path: str
    manifest_json_sha256: str  # MODEL-1's own convenience hash of the whole manifest file (NOT an aggregate freeze hash)
    n_files_checked: int
    n_violations: int
    violations: list[FileCheckResult]
    all_ok: bool
    checked_at_utc: str


def verify_v1_freeze(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> V1FreezeVerificationResult:
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    manifest_json_sha256 = canonical_sha256(manifest)

    results: list[FileCheckResult] = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        expected_sha256 = entry["sha256"]
        try:
            actual_bytes = path.read_bytes()
            actual_sha256 = hashlib.sha256(actual_bytes).hexdigest()
            match = actual_sha256 == expected_sha256
            results.append(
                FileCheckResult(path=str(path), manifest_sha256=expected_sha256, recomputed_sha256=actual_sha256, match=match)
            )
        except OSError as exc:
            results.append(
                FileCheckResult(
                    path=str(path), manifest_sha256=expected_sha256, recomputed_sha256=None, match=False, error=str(exc)
                )
            )

    violations = [r for r in results if not r.match]
    return V1FreezeVerificationResult(
        manifest_path=str(manifest_path),
        manifest_json_sha256=manifest_json_sha256,
        n_files_checked=len(results),
        n_violations=len(violations),
        violations=violations,
        all_ok=not violations,
        checked_at_utc=datetime.now(UTC).isoformat(),
    )


def write_operator_signing_request(result: V1FreezeVerificationResult, out_path: Path) -> None:
    """Emit an UNSIGNED operator-signing request, mirroring the manifest's
    own `signature_block` placeholder pattern. NEVER signed by MODEL-1."""
    doc = {
        "request_type": "v1_freeze_reverification",
        "manifest_path": result.manifest_path,
        "manifest_json_sha256": result.manifest_json_sha256,
        "n_files_checked": result.n_files_checked,
        "all_ok": result.all_ok,
        "n_violations": result.n_violations,
        "checked_at_utc": result.checked_at_utc,
        "signed_by": None,
        "signed_at": None,
        "signature": None,
        "note": "operator to sign (e.g. ssh-keygen -Y sign over this file's own hash) -- MODEL-1 holds no signing key and never signs this itself",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
