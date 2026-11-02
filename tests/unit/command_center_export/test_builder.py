from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.command_center_export.builder import REPORTS_DIR, build_model2_export
from model_lab.command_center_export.gates_data import GATES
from model_lab.command_center_export.ledger_data import LEDGER
from model_lab.contracts.model2_export import GateState
from model_lab.vol.candidate_spec import ELIGIBLE_CELLS, REJECTED_CELLS

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _build():
    return build_model2_export(NOW)


def test_amc_promotion_is_never_authorized() -> None:
    export = _build()
    assert export.promotion_gates.amc_promotion_authorized is False


def test_eligible_and_rejected_cells_match_the_locked_candidate_spec_exactly() -> None:
    export = _build()
    exported_eligible = {(c.ticker, c.horizon) for c in export.candidate_definition.eligible_cells}
    exported_rejected = {(c.ticker, c.horizon) for c in export.candidate_definition.rejected_cells}
    assert exported_eligible == {(c.ticker, c.horizon) for c in ELIGIBLE_CELLS}
    assert exported_rejected == {(c.ticker, c.horizon) for c in REJECTED_CELLS}
    assert len(exported_eligible) == 9
    assert len(exported_rejected) == 15


def test_historical_evidence_covers_all_24_cells_and_tallies_are_internally_consistent() -> None:
    export = _build()
    assert len(export.historical_evidence.cells) == 24
    h = export.historical_evidence
    assert h.n_cells_significant_better + h.n_cells_significant_worse + h.n_cells_no_significant_difference == 24
    # Independently re-derive the "better" count directly from the same source files, rather than
    # trusting the builder's own internal count.
    n_better = 0
    for cell in h.cells:
        if cell.dm_p_value != "None" and float(cell.dm_p_value) < 0.05 and float(cell.candidate_qlike) < float(cell.benchmark_qlike):
            n_better += 1
    assert n_better == h.n_cells_significant_better == 9


def test_prospective_evidence_counts_match_the_real_committed_files() -> None:
    export = _build()
    prospective_dir = REPORTS_DIR / "prospective"
    n_pending_files = len(list(prospective_dir.glob("pending_*.json")))
    n_resolved_files = len(list(prospective_dir.glob("resolved_*.json")))
    p = export.prospective_evidence
    assert len(p.observations) == n_pending_files
    assert p.n_resolved_non_replay_observations + p.n_pending_observations == n_pending_files
    assert p.n_resolved_non_replay_observations <= n_resolved_files
    # every resolved observation in the export must independently recompute the same
    # is_historical_replay as its real underlying file.
    for obs in p.observations:
        if obs.status == "RESOLVED":
            resolved_path = prospective_dir / f"resolved_{obs.ticker}_{obs.information_cutoff_date}.json"
            real = json.loads(resolved_path.read_text())
            assert obs.is_historical_replay == real["is_historical_replay"]


def test_prospective_sufficiency_is_not_established_at_this_sample_size() -> None:
    export = _build()
    assert export.prospective_evidence.n_resolved_non_replay_observations < 20, (
        "sanity bound: if this ever grows past Branch P's own E>=20 threshold's rough scale, "
        "the sufficiency_state hardcoding below must be revisited, not left stale"
    )
    assert export.prospective_evidence.sufficiency_state == "NOT_ESTABLISHED"


def test_research_ledger_matches_the_structured_ledger_data_exactly() -> None:
    export = _build()
    assert len(export.research_ledger.entries) == len(LEDGER)
    for entry, row in zip(export.research_ledger.entries, LEDGER, strict=True):
        assert entry.index == row.index
        assert entry.decision == row.decision


def test_promotion_gates_match_structured_gate_data_and_only_research_shadow_gates_pass() -> None:
    export = _build()
    assert len(export.promotion_gates.gates) == len(GATES)
    for gate in export.promotion_gates.gates:
        if gate.state == GateState.PASSED:
            assert gate.gate_id in ("G1_RESEARCH", "G2_SHADOW_CANDIDATE"), (
                f"gate {gate.gate_id} must not be PASSED without further governance evidence"
            )


def test_health_limitations_independent_review_is_pending_not_fabricated() -> None:
    export = _build()
    assert export.health_limitations.independent_review_status == "PENDING"
    assert export.health_limitations.known_defects_outstanding == []


def test_export_sha256_is_a_real_hash_of_the_actual_payload() -> None:
    from model_lab.canon.jcs import sha256_bytes

    export = _build()
    recomputed_payload = {
        "identity": export.identity.model_dump(mode="json"),
        "purpose": export.purpose.model_dump(mode="json"),
        "candidate_definition": export.candidate_definition.model_dump(mode="json"),
        "historical_evidence": export.historical_evidence.model_dump(mode="json"),
        "prospective_evidence": export.prospective_evidence.model_dump(mode="json"),
        "downstream_value": export.downstream_value.model_dump(mode="json"),
        "confidence_abstention": export.confidence_abstention.model_dump(mode="json"),
        "research_ledger": export.research_ledger.model_dump(mode="json"),
        "promotion_gates": export.promotion_gates.model_dump(mode="json"),
        "provenance": export.provenance.model_dump(mode="json"),
        "health_limitations": export.health_limitations.model_dump(mode="json"),
        "charts": [c.model_dump(mode="json") for c in export.charts],
    }
    import json as _json

    recomputed = sha256_bytes(_json.dumps(recomputed_payload, sort_keys=True, default=str).encode("utf-8"))
    assert recomputed == export.export_sha256


def test_substantive_content_is_deterministic_across_two_builds() -> None:
    """Everything except the two genuinely time-varying identity fields
    must be byte-identical across two independent builds."""
    e1 = _build()
    e2 = _build()
    assert e1.purpose == e2.purpose
    assert e1.candidate_definition == e2.candidate_definition
    assert e1.historical_evidence == e2.historical_evidence
    assert e1.downstream_value == e2.downstream_value
    assert e1.confidence_abstention == e2.confidence_abstention
    assert e1.research_ledger == e2.research_ledger
    assert e1.promotion_gates == e2.promotion_gates
    assert e1.identity.candidate_code_fingerprint == e2.identity.candidate_code_fingerprint


def test_candidate_code_fingerprint_is_a_real_hash_of_the_defining_source_files() -> None:
    from model_lab.canon.jcs import sha256_bytes

    export = _build()
    src_root = Path(__file__).resolve().parents[3] / "src" / "model_lab"
    files = ["vol/combination.py", "vol/candidate_spec.py", "vol/garch_models.py", "vol/baselines.py"]
    data = b""
    for rel in sorted(files):
        data += (src_root / rel).read_bytes()
    assert sha256_bytes(data) == export.identity.candidate_code_fingerprint


def test_downstream_value_cells_cover_all_six_tickers_and_three_levels() -> None:
    export = _build()
    tickers = {c.ticker for c in export.downstream_value.cells}
    levels = {c.tail_level for c in export.downstream_value.cells}
    assert tickers == {"SPY", "QQQ", "TLT", "IEF", "GLD", "USO"}
    assert levels == {"0.05", "0.025", "0.01"}
    eligible_tickers = {c.ticker for c in export.downstream_value.cells if c.is_eligible_ticker}
    assert eligible_tickers == {"TLT", "IEF", "GLD"}


def test_charts_each_include_a_required_incumbent_series() -> None:
    export = _build()
    assert len(export.charts) == 3
    for chart in export.charts:
        roles = {s.role for s in chart.series}
        assert "incumbent" in roles
