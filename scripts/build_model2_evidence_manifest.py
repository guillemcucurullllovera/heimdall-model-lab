#!/usr/bin/env python3
"""MODEL-2 evidence reconciliation (Phase 1 of the post-MODEL-2G
non-review work): walks every committed MODEL-2 result artifact under
`reports/` and builds one canonical manifest -- exact file path, SHA-256 of
its actual on-disk bytes, and the identity/config fields pulled directly out
of it (dataset_id, data_fingerprint, model configs/fingerprints, fold
counts, horizon, headline result fields).

This is ordinary implementation reconciliation, not an independent review:
its purpose is that every headline number quoted anywhere in the MODEL-2
reports/capstone has a traceable, re-hashable underlying artifact, and any
file that was supposed to be consistent with another (e.g. the same
dataset_id/data_fingerprint reported for a ticker across different
experiment families) can be mechanically cross-checked rather than asserted.

Run: `uv run python scripts/build_model2_evidence_manifest.py`
Output: `reports/MODEL2_EVIDENCE_MANIFEST.json`
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

# Each entry: (family_name, glob_pattern_relative_to_reports, identity_fields_to_extract)
EXPERIMENT_FAMILIES: list[tuple[str, str, list[str]]] = [
    ("vol_ladder", "vol_ladder/*.json", ["ticker", "horizon", "dataset_id", "data_fingerprint", "n_folds", "n_oos_total", "model_configs", "model_fingerprints", "no_leakage_proof"]),
    ("tail_ladder", "tail_ladder/*.json", ["ticker", "dataset_id", "data_fingerprint", "n_folds"]),
    ("anomaly_ladder", "anomaly_ladder/*.json", ["tickers", "data_fingerprints", "model_fingerprints", "n_folds", "n_oos_total", "caveat"]),
    ("changepoint_ladder", "changepoint_ladder/*.json", []),
    ("regime_value_add", "regime_value_add/*.json", ["ticker", "dataset_id", "data_fingerprint", "n_folds", "n_converged_folds", "n_well_identified_folds", "n_oos", "qlike_by_model"]),
    ("vol_model_comparison", "vol_model_comparison/*.json", ["ticker", "horizon", "n_oos", "model_ids", "spa_benchmark", "spa_p_value_consistent", "mcs_superior_set"]),
    ("regime_conditional_value", "regime_conditional_value/*.json", []),
    ("changepoint_gated_model_choice", "changepoint_gated_model_choice/*.json", []),
    ("shadow", "shadow/*.json", []),
]


def _git_head() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_identity(payload: Any, fields: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {f: payload[f] for f in fields if f in payload}


def build_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_name": "Model2EvidenceManifest",
        "schema_version": 1,
        "git": _git_head(),
        "families": {},
        "cross_checks": {},
    }

    dataset_id_by_ticker: dict[str, set[str]] = {}
    fingerprint_by_ticker: dict[str, set[str]] = {}

    for family_name, pattern, fields in EXPERIMENT_FAMILIES:
        files_out = []
        for path in sorted(REPORTS_DIR.glob(pattern)):
            if path.name == "_summary.json" and family_name not in ("changepoint_ladder",):
                # summaries are derived from the per-cell files already listed; still hash them,
                # but don't re-extract identity fields (they're aggregates, not identity).
                files_out.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)})
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = None
            entry: dict[str, Any] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)}
            identity = _extract_identity(payload, fields)
            if identity:
                entry["identity"] = identity
            files_out.append(entry)

            if isinstance(payload, dict):
                ticker = payload.get("ticker")
                ds_id = payload.get("dataset_id")
                fp = payload.get("data_fingerprint")
                if ticker and ds_id:
                    dataset_id_by_ticker.setdefault(ticker, set()).add(ds_id)
                if ticker and fp:
                    fingerprint_by_ticker.setdefault(ticker, set()).add(fp)

        manifest["families"][family_name] = {"pattern": pattern, "n_files": len(files_out), "files": files_out}

    # Cross-check: every experiment family that reports a given ticker must agree on the SAME
    # dataset_id and data_fingerprint for that ticker -- otherwise two "SPY" results could
    # silently be scored against two different underlying datasets.
    inconsistent_dataset_ids = {t: sorted(v) for t, v in dataset_id_by_ticker.items() if len(v) > 1}
    inconsistent_fingerprints = {t: sorted(v) for t, v in fingerprint_by_ticker.items() if len(v) > 1}
    manifest["cross_checks"]["dataset_id_consistent_per_ticker"] = not inconsistent_dataset_ids
    manifest["cross_checks"]["dataset_id_inconsistencies"] = inconsistent_dataset_ids
    manifest["cross_checks"]["data_fingerprint_consistent_per_ticker"] = not inconsistent_fingerprints
    manifest["cross_checks"]["data_fingerprint_inconsistencies"] = inconsistent_fingerprints
    manifest["cross_checks"]["tickers_covered"] = sorted(dataset_id_by_ticker.keys())

    return manifest


def main() -> None:
    manifest = build_manifest()
    out_path = REPORTS_DIR / "MODEL2_EVIDENCE_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    n_files = sum(f["n_files"] for f in manifest["families"].values())
    print(f"wrote {out_path} ({n_files} artifacts hashed across {len(manifest['families'])} families)")
    if not manifest["cross_checks"]["dataset_id_consistent_per_ticker"]:
        print("WARNING: dataset_id inconsistencies found:", manifest["cross_checks"]["dataset_id_inconsistencies"])
    if not manifest["cross_checks"]["data_fingerprint_consistent_per_ticker"]:
        print("WARNING: data_fingerprint inconsistencies found:", manifest["cross_checks"]["data_fingerprint_inconsistencies"])
    if manifest["cross_checks"]["dataset_id_consistent_per_ticker"] and manifest["cross_checks"]["data_fingerprint_consistent_per_ticker"]:
        print("Cross-check PASSED: every ticker's dataset_id and data_fingerprint agree across every experiment family.")


if __name__ == "__main__":
    main()
