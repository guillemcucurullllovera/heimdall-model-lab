"""PIT as-of query contract — P1-4 remediation (adversarial review finding
D.4): "There is no PIT query engine in the codebase yet (no `as_of`/
`available_at` query function exists), so this doesn't leak *today*, but the
'safe to use historically' framing rests on an unstated, unenforced domain
assumption... If the system has no historical query engine yet, encode the
contract now so future callers cannot silently misuse it."

This is the first `as_of` query function in MODEL-1. Its whole purpose is to
make the SNAPSHOT_ONLY misuse the review warned about structurally
impossible: a `SNAPSHOT_ONLY` series (§9 `pit_class`) has only ONE vintage —
whatever FRED/the vendor returned as "current" at fetch time — so a query
for what was knowable at some meaningfully earlier historical date CANNOT be
honestly answered from it. `query_as_of` refuses (returns `UNSAFE_...`,
never a fabricated value) rather than silently returning the current value
as if it had been known back then.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

from model_lab.contracts.dataset_snapshot import PitClass, PitRow

# How far in the past an `as_of` query may reach for a SNAPSHOT_ONLY series
# before being refused. A query within this window of "now" is treated as
# "effectively asking for the current value," which a single-vintage series
# CAN honestly answer; anything further back cannot. Conservative by design
# -- widen only with a documented domain justification, never silently.
SNAPSHOT_ONLY_SAFE_HORIZON = timedelta(days=5)


class AsOfStatus(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN_BEFORE_KNOWLEDGE_TIME = "UNKNOWN_BEFORE_KNOWLEDGE_TIME"
    UNSAFE_SNAPSHOT_ONLY_HISTORICAL_QUERY = "UNSAFE_SNAPSHOT_ONLY_HISTORICAL_QUERY"
    AS_REVISED_NOT_PIT_SAFE = "AS_REVISED_NOT_PIT_SAFE"
    NO_MATCHING_SERIES = "NO_MATCHING_SERIES"


@dataclass(frozen=True)
class AsOfResult:
    status: AsOfStatus
    value: str | None
    row: PitRow | None
    detail: str


def query_as_of(
    rows: list[PitRow],
    *,
    series_id: str,
    observation_period: date,
    as_of: datetime,
    pit_class: PitClass,
    retrieved_at_utc: datetime,
    snapshot_only_safe_horizon: timedelta = SNAPSHOT_ONLY_SAFE_HORIZON,
) -> AsOfResult:
    """What was knowable about `series_id`'s `observation_period` value, as
    of `as_of`? Never fabricates an answer: every non-KNOWN status carries
    `value=None`.

    - `AS_REVISED`: refused outright (§9 rule 5: "AS_REVISED series are
      flagged and excluded from any PIT-required task") -- this whole
      pit_class is not meant to answer as-of queries at all.
    - `SNAPSHOT_ONLY`: refused UNLESS `as_of` is within
      `snapshot_only_safe_horizon` of `retrieved_at_utc` (i.e. the query is
      effectively "what is the current value," which a single-vintage
      series can honestly answer).
    - `AS_RELEASED`: finds the vintage that was current as of `as_of` --
      the matching row with the latest `knowledge_time_utc <= as_of`. If
      `as_of` predates every row's `knowledge_time_utc`, the observation
      was simply not yet knowable.
    """
    if pit_class == PitClass.AS_REVISED:
        return AsOfResult(
            AsOfStatus.AS_REVISED_NOT_PIT_SAFE, None, None,
            "AS_REVISED series are excluded from PIT-required queries entirely (§9 rule 5)",
        )

    matching = [r for r in rows if r.series_id == series_id and r.observation_period == observation_period]
    if not matching:
        return AsOfResult(AsOfStatus.NO_MATCHING_SERIES, None, None, f"no row for {series_id}/{observation_period}")

    if pit_class == PitClass.SNAPSHOT_ONLY:
        if as_of < retrieved_at_utc - snapshot_only_safe_horizon:
            return AsOfResult(
                AsOfStatus.UNSAFE_SNAPSHOT_ONLY_HISTORICAL_QUERY, None, None,
                f"this series has only one vintage (fetched {retrieved_at_utc.isoformat()}); a query for "
                f"{as_of.isoformat()} ({(retrieved_at_utc - as_of).days} days earlier) cannot be honestly "
                "answered -- the current value is NOT evidence of what was knowable that far in the past "
                "(P1-4, adversarial review finding D.4)",
            )
        # Within the safe horizon: the single stored vintage IS the answer.
        row = matching[0]
        return AsOfResult(AsOfStatus.KNOWN, row.value, row, "SNAPSHOT_ONLY series, query within the safe horizon of the fetch time")

    # AS_RELEASED: genuine PIT semantics -- pick the latest vintage known by `as_of`.
    eligible = [r for r in matching if r.knowledge_time_utc <= as_of]
    if not eligible:
        earliest_known = min(r.knowledge_time_utc for r in matching)
        return AsOfResult(
            AsOfStatus.UNKNOWN_BEFORE_KNOWLEDGE_TIME, None, None,
            f"this observation was not knowable until {earliest_known.isoformat()}, after the "
            f"queried as_of={as_of.isoformat()}",
        )
    best = max(eligible, key=lambda r: r.knowledge_time_utc)
    return AsOfResult(AsOfStatus.KNOWN, best.value, best, f"vintage known as of {best.knowledge_time_utc.isoformat()}")


def utc_now() -> datetime:
    return datetime.now(UTC)
