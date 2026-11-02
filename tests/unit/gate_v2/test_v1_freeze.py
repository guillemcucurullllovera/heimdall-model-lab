from __future__ import annotations

from pathlib import Path

from model_lab.gate_v2.v1_freeze import verify_v1_freeze, write_operator_signing_request


def test_v1_freeze_verifies_against_the_real_manifest() -> None:
    result = verify_v1_freeze()
    assert result.n_files_checked == 33
    assert result.all_ok, f"unexpected v1 freeze violations: {result.violations}"
    assert len(result.manifest_json_sha256) == 64


def test_write_operator_signing_request_is_unsigned(tmp_path: Path) -> None:
    result = verify_v1_freeze()
    out = tmp_path / "request.json"
    write_operator_signing_request(result, out)
    import json

    doc = json.loads(out.read_text())
    assert doc["signed_by"] is None
    assert doc["signature"] is None
    assert doc["all_ok"] is True
