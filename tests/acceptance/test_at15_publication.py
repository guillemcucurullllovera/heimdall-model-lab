"""AT-15: manifest hash recompute; tampered file => UNAVAILABLE; ChartSpec
schema validation.

**P1-1 remediation** (adversarial review finding D.1): "`verify_publication`
only detects file/manifest *drift*, not a consistently-forged pair... A
coordinated edit of a content file *and* its corresponding `file_hashes`
entry together would not be caught." The tests in the "P1-1" section below
reproduce this exact gap and prove `verify_publication_authenticity`
(publish/anchor.py, reusing the P0-3 checkpoint primitive) closes it: a
coordinated rewrite changes `manifest.json`'s own bytes, so it no longer
matches an operator-signed checkpoint taken over the manifest -- caught as
`TAIL_TRUNCATED`, and `is_promotion_grade` becomes False. An
unanchored (no checkpoint yet) publication is explicitly `UNANCHORED`, not
silently treated as either valid or invalid -- it is honestly downgraded
to research-grade only."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from model_lab.anchor.contracts import AnchorStatus
from model_lab.contracts.chart_spec import (
    Causality,
    ChartSeries,
    ChartSeriesPoint,
    ChartSpec,
    DataAuthority,
)
from model_lab.contracts.evidence_report import EvidenceReport, ScoreResult
from model_lab.governance.signer_registry import RegisteredSigner, SignerRegistry, SignerRole
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    write_allowed_signers,
)
from model_lab.publish.anchor import create_publication_checkpoint, verify_publication_authenticity
from model_lab.publish.builder import build_publication
from model_lab.publish.verify import verify_publication

pytestmark = pytest.mark.acceptance
NAMESPACE = "model-lab-checkpoint"


def _chart() -> ChartSpec:
    return ChartSpec(
        chart_id="qlike-h1",
        decision_question="Does GJR-t beat EWMA(0.94) on 1-day-ahead QLIKE?",
        incumbent_id="EWMA_0.94",
        series=[
            ChartSeries(
                label="EWMA(0.94)",
                role="incumbent",
                points=[ChartSeriesPoint(x_label="2026", value="0.041")],
            ),
            ChartSeries(
                label="GJR-t",
                role="challenger",
                points=[ChartSeriesPoint(x_label="2026", unavailable=True)],
            ),
        ],
        n=2520,
        n_independent=252.0,
        mde="4.1%",
        data_cutoff=date(2026, 9, 30),
        causality=Causality.NOT_APPLICABLE,
        data_authority=DataAuthority.PROVISIONAL,
        chart_sha256="a" * 64,
    )


def _evidence_report() -> EvidenceReport:
    return EvidenceReport(
        report_id="er-1",
        model_version_hash="b" * 64,
        task_id="T-VOL",
        prereg_id="pr-1",
        run_id="run-1",
        as_of_utc=datetime(2026, 9, 30, tzinfo=UTC),
        primary_score_result=ScoreResult(name="QLIKE", value="0.038", n=2520, n_independent=252.0),
        report_sha256="c" * 64,
    )


def test_chartspec_must_include_an_incumbent_series() -> None:
    with pytest.raises(Exception, match="incumbent"):
        ChartSpec(
            chart_id="bad",
            decision_question="x",
            incumbent_id="EWMA",
            series=[ChartSeries(label="x", role="challenger", points=[])],
            causality=Causality.NOT_APPLICABLE,
            data_authority=DataAuthority.PROVISIONAL,
            chart_sha256="a" * 64,
        )


def test_unavailable_point_never_carries_a_value() -> None:
    with pytest.raises(Exception, match="unavailable"):
        ChartSeriesPoint(x_label="2026", value="0", unavailable=True)


def test_publication_manifest_hash_recomputes_on_untampered_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    manifest = build_publication(
        model_id="m1",
        model_version_hash="b" * 64,
        as_of="2026-09-30",
        ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(),
        charts=[_chart()],
        model_card={"family": "GJR-t"},
        datasheet={"source": "synthetic"},
    )
    out_dir = tmp_path / "m1" / ("b" * 64) / "2026-09-30"
    result = verify_publication(out_dir)
    assert result.ok
    assert result.data_authority == DataAuthority.PROVISIONAL
    assert manifest.data_authority == DataAuthority.PROVISIONAL


def test_tampered_file_renders_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    build_publication(
        model_id="m1",
        model_version_hash="b" * 64,
        as_of="2026-09-30",
        ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(),
        charts=[_chart()],
        model_card={"family": "GJR-t"},
        datasheet={"source": "synthetic"},
    )
    out_dir = tmp_path / "m1" / ("b" * 64) / "2026-09-30"
    chart_path = out_dir / "charts" / "qlike-h1.json"
    tampered = chart_path.read_text(encoding="utf-8").replace("0.041", "0.999")
    chart_path.write_text(tampered, encoding="utf-8")

    result = verify_publication(out_dir)
    assert not result.ok
    assert result.data_authority == DataAuthority.UNAVAILABLE
    assert "charts/qlike-h1.json" in result.mismatched_files


def test_missing_file_renders_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    build_publication(
        model_id="m1",
        model_version_hash="b" * 64,
        as_of="2026-09-30",
        ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(),
        charts=[_chart()],
        model_card={"family": "GJR-t"},
        datasheet={"source": "synthetic"},
    )
    out_dir = tmp_path / "m1" / ("b" * 64) / "2026-09-30"
    (out_dir / "datasheet.json").unlink()

    result = verify_publication(out_dir)
    assert not result.ok
    assert result.data_authority == DataAuthority.UNAVAILABLE
    assert "datasheet.json" in result.missing_files


# --- P1-1: publication manifest authenticity (checkpoint-anchored) --------


def _operator_registry_and_signers(tmp_path: Path):
    try:
        key = generate_throwaway_keypair(tmp_path / "operator-key")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available in this environment")
    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("op-1", [NAMESPACE], key.public_key_text)])
    registry = SignerRegistry(
        signers=[RegisteredSigner(principal="op-1", roles=[SignerRole.OPERATOR], namespaces=[NAMESPACE], public_key_line=key.public_key_text)]
    )
    return key, allowed_signers, registry


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("model_lab.publish.builder.PUBLISH_ROOT", tmp_path)
    build_publication(
        model_id="m1", model_version_hash="b" * 64, as_of="2026-09-30", ledger_head_hash="d" * 64,
        evidence_report=_evidence_report(), charts=[_chart()], model_card={"family": "GJR-t"},
        datasheet={"source": "synthetic"},
    )
    return tmp_path / "m1" / ("b" * 64) / "2026-09-30"


def test_unanchored_publication_is_explicitly_not_promotion_grade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = _build(tmp_path, monkeypatch)
    registry = SignerRegistry(signers=[])
    result = verify_publication_authenticity(out_dir, signer_registry=registry, allowed_signers_path=tmp_path / "unused")
    assert result.anchor.status == AnchorStatus.UNANCHORED
    assert result.is_promotion_grade is False
    # but the ordinary file-integrity check still passes -- research-usable, just not promotion-grade
    assert result.file_check.ok


def test_anchored_untampered_publication_is_promotion_grade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = _build(tmp_path, monkeypatch)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_publication_checkpoint(out_dir, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    result = verify_publication_authenticity(out_dir, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.anchor.status == AnchorStatus.ANCHORED_VALID
    assert result.is_promotion_grade is True


def test_EXPLOIT_coordinated_file_and_manifest_rewrite_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's exact gap: edit a content file AND its file_hashes
    entry together, keeping internal (verify_publication) consistency --
    the OLD check alone would miss this. The anchor check must catch it."""
    out_dir = _build(tmp_path, monkeypatch)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_publication_checkpoint(out_dir, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    # Coordinated tamper: change the chart content AND recompute its hash
    # in the manifest, so verify_publication (file/manifest drift only)
    # would see a perfectly self-consistent pair.
    chart_path = out_dir / "charts" / "qlike-h1.json"
    tampered_content = chart_path.read_text(encoding="utf-8").replace("0.041", "0.999")
    chart_path.write_text(tampered_content, encoding="utf-8")

    from model_lab.canon.jcs import sha256_bytes

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_hashes"]["charts/qlike-h1.json"] = sha256_bytes(tampered_content.encode("utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The OLD check alone is fooled (self-consistent forged pair):
    old_check = verify_publication(out_dir)
    assert old_check.ok, "sanity: the coordinated forgery IS internally self-consistent"

    # The anchored check is NOT fooled -- manifest.json's own bytes changed.
    result = verify_publication_authenticity(out_dir, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.anchor.status == AnchorStatus.TAIL_TRUNCATED
    assert result.is_promotion_grade is False


def test_forged_publication_checkpoint_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = _build(tmp_path, monkeypatch)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_publication_checkpoint(out_dir, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    cp_path = out_dir / "manifest.json.checkpoints.ndjson"
    cp_line = json.loads(cp_path.read_text(encoding="utf-8").splitlines()[0])
    cp_line["checkpoint"]["anchored_head_hash"] = "f" * 64  # forged
    cp_path.write_text(json.dumps(cp_line) + "\n", encoding="utf-8")

    result = verify_publication_authenticity(out_dir, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.anchor.status == AnchorStatus.CHECKPOINT_INVALID
    assert result.is_promotion_grade is False
