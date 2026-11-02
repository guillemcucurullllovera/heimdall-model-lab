from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from model_lab.contracts.licence import LicenceStatus
from model_lab.contracts.option_snapshot import (
    AcquisitionStatus,
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

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _record(seq: int, prev_hash: str | None) -> OptionSnapshotRecord:
    draft = OptionSnapshotRecord(
        archive_seq=seq, prev_hash=prev_hash, record_hash="0" * 64,
        observation_utc=NOW, observation_kind=ObservationKind.AMC_DISCOVERY,
        provider_id="alpaca", provider_feed="indicative",
        response_fingerprint="a" * 64, normalized_rows_sha256="b" * 64,
        acquisition_status=AcquisitionStatus.OK,
        licence=OptionLicenceInfo(status=LicenceStatus.UNRESOLVED, checked_utc=NOW, permitted_use_note="test"),
        underlying=UnderlyingObservation(symbol="SPY", price="500", price_source="alpaca", price_time_utc=NOW),
        contract=OptionContractSpec(occ_symbol="SPY260101C00500000", expiry=date(2026, 1, 1), strike="500", right=OptionRight.CALL, style=ExerciseStyle.AMERICAN),
        quote=OptionQuote(bid="1.0", ask="1.1"),
        mid=MidDerivation(mid="1.05", derived=True, crossed=False),
        market_state=MarketState(),
        raw_ref=RawRef(content_addressed_path="/tmp/x", byte_length=10),
        replay_grade=ReplayGrade.FETCH_TIME_ONLY,
    )
    real_hash = record_hash_of(draft)
    return draft.model_copy(update={"record_hash": real_hash})


def test_next_seq_and_prev_hash_empty() -> None:
    assert next_seq_and_prev_hash([]) == (0, None)


def test_next_seq_and_prev_hash_continues_chain() -> None:
    r0 = _record(0, None)
    seq, prev_hash = next_seq_and_prev_hash([r0])
    assert seq == 1
    assert prev_hash == r0.record_hash


def test_record_hash_of_excludes_the_record_hash_field_itself() -> None:
    r = _record(0, None)
    other = r.model_copy(update={"record_hash": "f" * 64})
    assert record_hash_of(r) == record_hash_of(other)


def test_record_hash_of_changes_with_content() -> None:
    r0 = _record(0, None)
    r1 = _record(1, "a" * 64)
    assert record_hash_of(r0) != record_hash_of(r1)


def test_archive_file_append_many_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "archive.ndjson"
    archive = OptionArchiveFile(path)
    assert archive.read_all() == []

    def _build(existing: list[OptionSnapshotRecord]) -> list[OptionSnapshotRecord]:
        seq, prev = next_seq_and_prev_hash(existing)
        return [_record(seq, prev)]

    archive.append_many(_build)
    archive.append_many(_build)

    records = archive.read_all()
    assert len(records) == 2
    assert records[0].archive_seq == 0
    assert records[1].archive_seq == 1
    assert records[1].prev_hash == records[0].record_hash
