"""Observer-side OptionSnapshotRecord ingester — §18a.

"the immediate action is an *observer-side ingester* that reads the AMC's
persisted candidate/validation files read-only." No new source is added; no
new polling; everything here reads exactly what the pinned AMC root's
`cycles/*_candidates.json` files (see `replay/amc_root.py` for the path)
already contain, confirmed this session:

- 22 candidate files total (5 for experiment `amc4-prospective-v1-20260908`,
  17 for `amc4-prospective-v2-20260925`); 20 of 22 are `[]`.
- The only two non-empty files are BOTH from the v1 experiment:
  `cycle-003-20260908_candidates.json` and `cycle-004-20260908_candidates.json`
  (736 candidates each: SPY 122 calls, GLD 257 calls, TLT 357 puts) — these
  are two DISTINCT observations ~3 minutes apart (as_of 22:59:41 vs
  23:02:25), NOT byte-identical duplicates of each other (an earlier
  cross-worktree duplicate exists elsewhere on the host, unrelated to this
  pair — verified directly before writing this module; do not assume
  dedup between cycle-003 and cycle-004).
- Each candidate is a multi-leg structure; §18a records one
  OptionSnapshotRecord PER CONTRACT (per leg), not per candidate structure.
- Fields actually present per leg (real schema, confirmed by direct read):
  `indicative_bid`, `indicative_ask`, `indicative_source_timestamp` (a
  vendor [Alpaca] quote timestamp — genuinely vendor-supplied, so
  `quote_time_utc` is legitimately set, not inferred), `occ_symbol`,
  `strike`, `expiry`, `right` ("C"/"P"), `multiplier`, `root`. Candidate-level
  `provisional_greeks` are the AMC's OWN solver output (Heimdall's own IV/
  greeks per R2), NOT vendor-documented — these go into `derived`, never
  `vendor_greeks`. No OI, no volume, no exchange flags anywhere in this
  data (confirmed absent).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.licence import LicenceStatus
from model_lab.contracts.option_snapshot import (
    AcquisitionStatus,
    DataQualityFlag,
    DerivedAnalytics,
    ExerciseStyle,
    MarketState,
    MidDerivation,
    ObservationKind,
    OptionContractSpec,
    OptionLicenceInfo,
    OptionQuote,
    OptionRight,
    OptionSnapshotRecord,
    RawRef,
    ReplayGrade,
    UnderlyingObservation,
)
from model_lab.options_archive.storage import (
    OptionArchiveFile,
    next_seq_and_prev_hash,
    record_hash_of,
)

RAW_STORE_ROOT = Path("./local-data/raw")
ARCHIVE_PATH = Path("./local-data/options_archive/archive.ndjson")

ZERO_HASH = "0" * 64  # placeholder for record_hash before self-referential compute

ALPACA_LICENCE_NOTE = (
    "Alpaca indicative options feed (vendor-computed/modeled quotes, NOT OPRA/NBBO); "
    "personal/non-commercial account terms per R1b; archived here only because the AMC "
    "already ingested and persisted it (operator-accepted internal-archive risk, §18a rule 6). "
    "research_only_unlicensed until firm-use rights are resolved in writing."
)

WIDE_SPREAD_FRACTION_OF_MID = 0.10  # R2: observed median spread 3.97%; flag well above typical


def _store_raw_file(path: Path) -> RawRef:
    data = path.read_bytes()
    digest = sha256_bytes(data)
    dest_dir = RAW_STORE_ROOT / digest[:2]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / digest
    if not dest.exists():
        dest.write_bytes(data)
    return RawRef(content_addressed_path=str(dest), byte_length=len(data))


def _dec(s: str | None) -> Decimal | None:
    if s is None:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _quality_flags(bid: Decimal | None, ask: Decimal | None, mid: Decimal | None) -> list[DataQualityFlag]:
    flags: list[DataQualityFlag] = []
    if bid is not None and ask is not None:
        if bid > ask:
            flags.append(DataQualityFlag.CROSSED)
        elif bid == ask:
            flags.append(DataQualityFlag.LOCKED)
    if bid is not None and bid <= 0:
        flags.append(DataQualityFlag.ZERO_BID)
    if bid is not None and ask is not None and mid is not None and mid > 0:
        spread_frac = (ask - bid) / mid
        if spread_frac > Decimal(str(WIDE_SPREAD_FRACTION_OF_MID)):
            flags.append(DataQualityFlag.WIDE_SPREAD)
    return flags


def _leg_to_record(
    *,
    candidate: dict,
    leg: dict,
    observation_kind: ObservationKind,
    amc_experiment_id: str,
    amc_cycle_id: str,
    response_fingerprint: str,
    raw_ref: RawRef,
) -> OptionSnapshotRecord:
    greeks = candidate.get("provisional_greeks") or {}

    bid = _dec(leg.get("indicative_bid"))
    ask = _dec(leg.get("indicative_ask"))
    crossed = bid is not None and ask is not None and bid > ask
    mid_val = None
    derived_mid = False
    if bid is not None and ask is not None and bid > 0 and ask > 0 and not crossed:
        mid_val = (bid + ask) / 2
        derived_mid = True

    quote_time = leg.get("indicative_source_timestamp")
    quote_time_dt = datetime.fromisoformat(quote_time) if quote_time else None

    contract = OptionContractSpec(
        occ_symbol=leg["occ_symbol"],
        expiry=leg["expiry"],
        strike=leg["strike"],
        right=OptionRight.CALL if leg["right"] == "C" else OptionRight.PUT,
        style=ExerciseStyle.AMERICAN,  # all six ETFs list American-exercise options
        multiplier=int(leg.get("multiplier", 100)),
    )

    underlying_price = greeks.get("underlying_price")
    underlying_time = greeks.get("underlying_source_timestamp")
    underlying_time_dt = (
        datetime.fromisoformat(underlying_time) if underlying_time else datetime.fromisoformat(candidate["as_of"])
    )
    price_age_s = None
    if quote_time_dt is not None and underlying_time_dt is not None:
        price_age_s = round(abs((underlying_time_dt - quote_time_dt).total_seconds()))

    quality_flags = _quality_flags(bid, ask, mid_val)
    if greeks.get("implied_volatility") is None:
        quality_flags.append(DataQualityFlag.MISSING_IV)
    iv = _dec(greeks.get("implied_volatility"))
    if iv is not None and (iv <= 0 or iv > 3):  # >300% IV treated as extreme
        quality_flags.append(DataQualityFlag.EXTREME_IV)
    if price_age_s is not None and price_age_s > 300:  # >5 minutes stale, per R2's observed up-to-51-min gaps
        quality_flags.append(DataQualityFlag.OFF_MARKET_UNDERLYING)

    draft = OptionSnapshotRecord(
        archive_seq=0,  # placeholder; assigned for real at append time
        prev_hash=None,  # placeholder; assigned for real at append time
        record_hash=ZERO_HASH,  # placeholder; computed below
        observation_utc=candidate["as_of"],
        observation_kind=observation_kind,
        amc_experiment_id=amc_experiment_id,
        amc_cycle_id=amc_cycle_id,
        provider_id=candidate.get("indicative_source_provider_id", "alpaca"),
        provider_feed=candidate.get("indicative_data_label", "indicative"),
        request_fingerprint=None,
        response_fingerprint=response_fingerprint,
        normalized_rows_sha256=sha256_bytes(json.dumps({"candidate": candidate["candidate_id"], "leg": leg}, sort_keys=True).encode()),
        acquisition_status=AcquisitionStatus.OK,
        licence=OptionLicenceInfo(
            status=LicenceStatus.UNRESOLVED,
            basis_url=None,
            checked_utc=datetime.now(UTC),
            permitted_use_note=ALPACA_LICENCE_NOTE,
        ),
        underlying=UnderlyingObservation(
            symbol=leg["root"],
            price=str(underlying_price) if underlying_price is not None else "0",
            price_source=greeks.get("underlying_source_provider_id", "alpaca"),
            price_time_utc=underlying_time_dt,
            price_age_s=price_age_s,
        ),
        contract=contract,
        quote=OptionQuote(
            bid=leg.get("indicative_bid"),
            ask=leg.get("indicative_ask"),
            quote_time_utc=quote_time_dt,
            quote_age_s=0 if quote_time_dt is not None else None,
            exchange_flags=[],
        ),
        mid=MidDerivation(
            mid=str(mid_val) if mid_val is not None else None, derived=derived_mid, crossed=crossed
        ),
        market_state=MarketState(),  # no OI/volume in this data, confirmed
        vendor_greeks=None,  # provisional_greeks are Heimdall's own, not vendor-documented
        derived=DerivedAnalytics(
            iv_mid_bs=greeks.get("implied_volatility"),
            iv_solver_version=None,
            rate_source="AMC flat-rate assumption",
            rate=greeks.get("risk_free_rate"),
            dividend_yield_or_schedule="0",
            dividend_source="AMC zero-dividend assumption",
            greeks_own={
                k: greeks[k]
                for k in ("delta", "gamma", "theta", "vega", "vanna", "volga")
                if greeks.get(k) is not None
            },
            american_adjustment_method="BS-European approximation (undocumented; per R2 finding)",
        ),
        data_quality_flags=quality_flags,
        raw_ref=raw_ref,
        replay_grade=(
            ReplayGrade.VENDOR_QUOTE_TIMESTAMPED if quote_time_dt is not None else ReplayGrade.FETCH_TIME_ONLY
        ),
    )
    return draft


def ingest_candidate_file(path: Path, *, amc_experiment_id: str, amc_cycle_id: str) -> list[OptionSnapshotRecord]:
    """Read one candidates.json file and return the OptionSnapshotRecord
    drafts (archive_seq/prev_hash/record_hash still placeholders — assigned
    by `finalize_and_append`)."""
    raw_ref = _store_raw_file(path)
    response_fingerprint = sha256_bytes(path.read_bytes())
    candidates = json.loads(path.read_text(encoding="utf-8"))

    drafts: list[OptionSnapshotRecord] = []
    for candidate in candidates:
        for leg in candidate.get("legs", []):
            drafts.append(
                _leg_to_record(
                    candidate=candidate,
                    leg=leg,
                    observation_kind=ObservationKind.AMC_DISCOVERY,
                    amc_experiment_id=amc_experiment_id,
                    amc_cycle_id=amc_cycle_id,
                    response_fingerprint=response_fingerprint,
                    raw_ref=raw_ref,
                )
            )
    return drafts


def finalize_and_append(archive: OptionArchiveFile, drafts: list[OptionSnapshotRecord]) -> list[OptionSnapshotRecord]:
    """Assign real archive_seq/prev_hash/record_hash to `drafts`, in order,
    continuing from the archive's current tail, and append them atomically."""

    def _build(existing: list[OptionSnapshotRecord]) -> list[OptionSnapshotRecord]:
        seq, prev_hash = next_seq_and_prev_hash(existing)
        finalized: list[OptionSnapshotRecord] = []
        for draft in drafts:
            with_chain_fields = draft.model_copy(update={"archive_seq": seq, "prev_hash": prev_hash})
            record_hash = record_hash_of(with_chain_fields)
            final = with_chain_fields.model_copy(update={"record_hash": record_hash})
            finalized.append(final)
            seq += 1
            prev_hash = record_hash
        return finalized

    return archive.append_many(_build)
