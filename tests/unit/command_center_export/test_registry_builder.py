from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.command_center_export.model_registry_data import STATIC_ENTRIES
from model_lab.command_center_export.registry_builder import REPORTS_DIR, build_model_lab_registry
from model_lab.contracts.model_lab_registry import ModelLabRegistryExport, ResearchState

NOW = datetime(2026, 10, 1, tzinfo=UTC)
EXPORT_PATH = Path(__file__).resolve().parents[3] / "reports" / "model_lab_registry" / "registry_export.json"


def _build():
    return build_model_lab_registry(NOW)


def test_committed_export_file_is_actually_loadable_against_the_contract() -> None:
    assert EXPORT_PATH.exists(), "run scripts/build_model_lab_registry.py before this test"
    loaded = ModelLabRegistryExport.model_validate_json(EXPORT_PATH.read_text())
    assert loaded.schema_name == "ModelLabRegistryExport"
    assert len(loaded.entries) == len(STATIC_ENTRIES)


def test_no_duplicate_model_ids() -> None:
    reg = _build()
    ids = [e.model_id for e in reg.entries]
    assert len(ids) == len(set(ids))


def test_amc_promotion_is_never_authorized_for_any_entry() -> None:
    reg = _build()
    for e in reg.entries:
        assert e.amc_promotion_authorized is False, f"{e.model_id}: amc_promotion_authorized must always be False"
    assert reg.research_summary.n_models_amc_eligible == 0


def test_all_five_research_states_are_actually_present_not_hidden() -> None:
    reg = _build()
    present_states = {e.research_state for e in reg.entries}
    assert present_states == set(ResearchState), "every one of the 5 required states must appear at least once"
    assert reg.research_summary.n_rejected_or_retired >= 1, "the rejected/retired category must not be hidden"
    assert reg.research_summary.n_literature_research_candidates >= 1


def test_task_hierarchy_and_status_buckets_only_reference_real_entries() -> None:
    reg = _build()
    real_ids = {e.model_id for e in reg.entries}
    for task_id, model_ids in reg.task_hierarchy.items():
        for mid in model_ids:
            assert mid in real_ids, f"task_hierarchy[{task_id}] references unknown model_id {mid}"
    for bucket, model_ids in reg.status_buckets.items():
        for mid in model_ids:
            assert mid in real_ids, f"status_buckets[{bucket}] references unknown model_id {mid}"


def test_model2_composite_constituents_are_bidirectionally_consistent() -> None:
    reg = _build()
    by_id = {e.model_id: e for e in reg.entries}
    composite = by_id["Combo_median_EWMA_GARCH_EGARCH"]
    assert set(composite.constituents) == {"EWMA_0.94", "GARCH_t", "EGARCH_t"}
    for constituent_id in composite.constituents:
        constituent = by_id[constituent_id]
        assert constituent.constituent_of == "Combo_median_EWMA_GARCH_EGARCH", (
            f"{constituent_id} must point back to the composite it is a constituent of"
        )
        # each constituent must be independently represented, not an opaque sub-field of the composite
        assert constituent.task_id == "T-VOL"
        assert constituent.historical_evidence_refs, f"{constituent_id} must have its own independent historical evidence"


def test_implemented_and_shadow_entries_expose_real_nonempty_outputs() -> None:
    reg = _build()
    for e in reg.entries:
        if e.research_state in (ResearchState.IMPLEMENTED_REPRODUCIBLE, ResearchState.IMPLEMENTED_VALIDATION_PENDING, ResearchState.SHADOW_RESEARCH, ResearchState.REJECTED_OR_RETIRED):
            assert e.actual_outputs_sample, f"{e.model_id} ({e.research_state}) must expose a real, non-empty output sample, not just a description"


def test_literature_candidates_expose_no_fabricated_evidence() -> None:
    reg = _build()
    for e in reg.entries:
        if e.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
            assert e.actual_outputs_sample == {}, f"{e.model_id}: a literature candidate must never carry a fabricated output sample"
            assert e.chart_ids == [], f"{e.model_id}: a literature candidate must never carry a chart (no evidence exists to plot)"
            assert e.code_fingerprint is None


def test_every_chart_id_referenced_by_an_entry_exists_in_the_charts_list() -> None:
    reg = _build()
    real_chart_ids = {c.chart_id for c in reg.charts}
    for e in reg.entries:
        for cid in e.chart_ids:
            assert cid in real_chart_ids, f"{e.model_id} references chart_id {cid!r} which is not in reg.charts"


def test_every_chart_has_a_required_incumbent_series() -> None:
    reg = _build()
    assert len(reg.charts) == 15
    for chart in reg.charts:
        roles = {s.role for s in chart.series}
        assert "incumbent" in roles, f"{chart.chart_id} has no incumbent series"


def test_vol_chart_values_match_the_real_committed_vol_ladder_json() -> None:
    reg = _build()
    chart = next(c for c in reg.charts if c.chart_id == "vol_qlike_by_horizon_TLT")
    garch_series = next(s for s in chart.series if s.label == "GARCH_t")
    real = json.loads((REPORTS_DIR / "vol_ladder" / "TLT_h1.json").read_text())
    h1_point = next(p for p in garch_series.points if p.x_label == "h=1")
    assert h1_point.value == str(real["results"]["GARCH_t"]["qlike_mean"])


def test_regime_chart_values_match_the_real_committed_regime_json() -> None:
    reg = _build()
    chart = next(c for c in reg.charts if c.chart_id == "regime_vs_garch_qlike")
    garch_series = next(s for s in chart.series if s.label == "GARCH_t")
    real = json.loads((REPORTS_DIR / "regime_value_add" / "TLT.json").read_text())
    tlt_point = next(p for p in garch_series.points if p.x_label == "TLT")
    assert tlt_point.value == str(real["qlike_by_model"]["GARCH_t"])


def test_export_sha256_is_a_real_hash_of_the_actual_payload() -> None:
    from model_lab.canon.jcs import sha256_bytes

    reg = _build()
    payload = {
        "generated_at_utc": str(reg.generated_at_utc), "git_commit": reg.git_commit, "git_branch": reg.git_branch,
        "entries": [e.model_dump(mode="json") for e in reg.entries],
        "task_hierarchy": reg.task_hierarchy, "status_buckets": reg.status_buckets,
        "amc_promotion_evaluations": {k: v.model_dump(mode="json") for k, v in reg.amc_promotion_evaluations.items()},
        "charts": [c.model_dump(mode="json") for c in reg.charts], "research_summary": reg.research_summary.model_dump(mode="json"),
    }
    recomputed = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    assert recomputed == reg.export_sha256


def test_substantive_content_is_deterministic_across_two_builds() -> None:
    r1 = _build()
    r2 = _build()
    assert r1.entries == r2.entries
    assert r1.charts == r2.charts
    assert r1.task_hierarchy == r2.task_hierarchy
    assert r1.status_buckets == r2.status_buckets
    assert r1.research_summary == r2.research_summary
    assert r1.amc_promotion_evaluations == r2.amc_promotion_evaluations


def test_research_summary_counts_match_actual_entry_tallies() -> None:
    reg = _build()
    s = reg.research_summary
    assert s.n_implemented_reproducible == sum(1 for e in reg.entries if e.research_state == ResearchState.IMPLEMENTED_REPRODUCIBLE)
    assert s.n_implemented_validation_pending == sum(1 for e in reg.entries if e.research_state == ResearchState.IMPLEMENTED_VALIDATION_PENDING)
    assert s.n_shadow_research == sum(1 for e in reg.entries if e.research_state == ResearchState.SHADOW_RESEARCH)
    assert s.n_literature_research_candidates == sum(1 for e in reg.entries if e.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE)
    assert s.n_rejected_or_retired == sum(1 for e in reg.entries if e.research_state == ResearchState.REJECTED_OR_RETIRED)
    total = s.n_implemented_reproducible + s.n_implemented_validation_pending + s.n_shadow_research + s.n_literature_research_candidates + s.n_rejected_or_retired
    assert total == len(reg.entries)
