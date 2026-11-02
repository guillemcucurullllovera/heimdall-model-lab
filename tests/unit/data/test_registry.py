from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.data.registry import (
    default_source_registry,
    get_entry,
    load_or_init_registry,
    save_registry,
)


def test_every_default_entry_has_mechanically_correct_research_state() -> None:
    for entry in default_source_registry():
        assert entry.derived_research_state_ok, f"{entry.source_id} research_state mismatch"


def test_alpaca_and_massive_are_unresolved_research_only() -> None:
    entries = {e.source_id: e for e in default_source_registry()}
    for sid in ("alpaca_bars", "massive_bars", "marketdata_app"):
        assert entries[sid].licence_status == LicenceStatus.UNRESOLVED
        assert entries[sid].research_state == ResearchState.RESEARCH_ONLY_UNLICENSED


def test_fred_is_clear_promotion_eligible() -> None:
    entries = {e.source_id: e for e in default_source_registry()}
    assert entries["fred_alfred"].licence_status == LicenceStatus.CLEAR
    assert entries["fred_alfred"].research_state == ResearchState.PROMOTION_ELIGIBLE


def test_cboe_yahoo_ice_are_restricted_and_never_queried() -> None:
    entries = {e.source_id: e for e in default_source_registry()}
    for sid in ("cboe_scripted", "yahoo_finance", "ice_hy_oas"):
        assert entries[sid].licence_status == LicenceStatus.RESTRICTED
        assert entries[sid].research_state == ResearchState.QUARANTINED
        assert entries[sid].queried is False


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "source_registry.json"
    entries = default_source_registry()
    save_registry(entries, path)
    loaded = load_or_init_registry(path)
    assert {e.source_id for e in loaded} == {e.source_id for e in entries}


def test_get_entry_unknown_raises(tmp_path: Path) -> None:
    path = tmp_path / "source_registry.json"
    save_registry(default_source_registry(), path)
    entries = load_or_init_registry(path)
    with pytest.raises(KeyError):
        get_entry("does_not_exist", entries)
