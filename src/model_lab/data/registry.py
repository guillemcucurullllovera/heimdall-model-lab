"""Source registry — MODEL1_data_readiness.md / operator item B: distinguish
CLEAR / UNRESOLVED / RESTRICTED sources, pre-populated from the R1b terms
audit findings already established in the readiness pack. Persisted to
`./local-data/source_registry.json`.

Authority is never upgraded silently: `research_state` on every downstream
DatasetSnapshot is mechanically derived from a source's `licence_status` via
`model_lab.contracts.licence.derive_research_state` — nothing in this module
or its callers hand-assigns a research state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel
from model_lab.contracts.licence import LicenceStatus, ResearchState, derive_research_state

DEFAULT_REGISTRY_PATH = Path("./local-data/source_registry.json")


class SourceRegistryEntry(ModelLabBaseModel):
    source_id: str
    provider: str
    description: str
    licence_status: LicenceStatus
    research_state: ResearchState
    basis: str = Field(description="why this status was assigned, citing the R1b/D3/D4 finding")
    queried: bool = Field(description="False for sources the estate has ruled out and MODEL-1 never calls")

    @property
    def derived_research_state_ok(self) -> bool:
        return self.research_state == derive_research_state(self.licence_status)


def _entry(
    source_id: str, provider: str, description: str, licence_status: LicenceStatus, basis: str, *, queried: bool = True
) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=source_id,
        provider=provider,
        description=description,
        licence_status=licence_status,
        research_state=derive_research_state(licence_status),
        basis=basis,
        queried=queried,
    )


def default_source_registry() -> list[SourceRegistryEntry]:
    return [
        _entry(
            "alpaca_bars",
            "Alpaca",
            "Alpaca market-data daily OHLCV bars (SIP), 6-ETF universe",
            LicenceStatus.UNRESOLVED,
            "D4 HOLD: personal/free-tier terms ('personal, non-commercial'); firm-use rights not "
            "confirmed in writing. Operator ruling 2026-09-30: existing personal credentials may be "
            "used for exploratory/internal research only; mark research_only_unlicensed.",
        ),
        _entry(
            "massive_bars",
            "Massive (Polygon-compatible)",
            "Massive daily OHLCV bars, 6-ETF universe; Basic free tier (~2y depth)",
            LicenceStatus.UNRESOLVED,
            "D4 HOLD: free-tier individual-use terms; options_aggregates_history entitlement "
            "confirmed NOT_AUTHORIZED on free tier.",
        ),
        _entry(
            "marketdata_app",
            "MarketData.app",
            "MarketData.app EOD option chains (date= param) and stock quotes",
            LicenceStatus.UNRESOLVED,
            "D4 HOLD: 'non-professional individuals only' on self-service plans; delete-data-on-"
            "termination clause conflicts with a durable archive. Starter Trial: no OPRA/IEX/UTP "
            "entitlement (confirmed live, HTTP 203 Non-Authoritative on every quote).",
        ),
        _entry(
            "fred_alfred",
            "FRED / ALFRED (St. Louis Fed)",
            "US government / central-bank macro series, PIT vintages via ALFRED",
            LicenceStatus.CLEAR,
            "Public-domain US government data. Operator ruling 2026-09-30 item 8 authorises the "
            "existing keyed FRED credential for Model Lab under its own process/rate-cap/cache.",
        ),
        _entry(
            "cboe_scripted",
            "Cboe (undocumented/scripted endpoint)",
            "Cboe SPX/VIX index-level scripted endpoint",
            LicenceStatus.RESTRICTED,
            "D3: REJECTED. Personal non-commercial terms, no storing. Operator ruling item 6: "
            "'Cboe undocumented/scripted endpoint remains REJECTED.' Never queried by MODEL-1.",
            queried=False,
        ),
        _entry(
            "yahoo_finance",
            "Yahoo Finance",
            "Yahoo SPX OHLC",
            LicenceStatus.RESTRICTED,
            "R1b: PROHIBITED for firm use; also has stale-open data quality defects. Operator "
            "ruling item 7: do not use Yahoo as canonical evidence. Never queried by MODEL-1.",
            queried=False,
        ),
        _entry(
            "ice_hy_oas",
            "ICE Data (BofA HY OAS)",
            "ICE BofA High Yield OAS credit spread",
            LicenceStatus.RESTRICTED,
            "R1b: PROHIBITED without ICE's written permission; FRED now limits the series to 3y. "
            "Never queried by MODEL-1.",
            queried=False,
        ),
        _entry(
            "baa10y",
            "Moody's / FRED (BAA10Y)",
            "Moody's Baa corporate bond yield spread",
            LicenceStatus.UNRESOLVED,
            "R1b: conflicting notices (FRED allows internal commercial use; Moody's terms bar "
            "'stored for subsequent use'). Not queried by MODEL-1 in this pass (not needed for the "
            "ETF/macro-PIT bootstrap); left UNRESOLVED for a future ruling.",
            queried=False,
        ),
    ]


def load_or_init_registry(path: Path = DEFAULT_REGISTRY_PATH) -> list[SourceRegistryEntry]:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [SourceRegistryEntry.model_validate(e) for e in raw["entries"]]
    entries = default_source_registry()
    save_registry(entries, path)
    return entries


def save_registry(entries: list[SourceRegistryEntry], path: Path = DEFAULT_REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_name": "SourceRegistry",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "entries": [e.canonical_dict() for e in entries],
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_entry(source_id: str, entries: list[SourceRegistryEntry] | None = None) -> SourceRegistryEntry:
    entries = entries if entries is not None else load_or_init_registry()
    for e in entries:
        if e.source_id == source_id:
            return e
    raise KeyError(f"unknown source_id {source_id!r}")
