"""Thin, explicitly-labelled wrapper around the already-authorised
`MassiveClient` (read-only, no purchases -- the same provider and the same
discipline MODEL-2A's own VIX-availability checks used this session) for
the ONE purpose this package needs it for: checking whether the real world
has produced a trading day beyond the frozen research dataset's last known
date, and if so, what that day's close actually was.

Never writes anywhere in the frozen data estate; never called by any
historical walk-forward evaluation; only ever called by `resolve.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from model_lab.data.providers._http import SourceUnavailableError
from model_lab.data.providers.massive import MassiveClient, MassiveError


@dataclass(frozen=True)
class LiveBar:
    trading_date: str  # ISO date
    close: str


def fetch_bars_after(ticker: str, since_date_iso: str, *, lookahead_days: int = 10) -> list[LiveBar]:
    """Real, read-only, live fetch of every bar strictly after
    `since_date_iso` up to `lookahead_days` calendar days later (bounded,
    not an unbounded fetch -- this is a prospective-evidence check, not a
    data-acquisition operation). Returns an EMPTY list, never raises, if
    the provider has no new bar yet (a quiet weekend/holiday, or the next
    trading day genuinely has not happened yet), OR the provider is
    transiently unavailable (rate-limited, network error) -- the caller
    (resolve.py) is responsible for leaving the observation PENDING in
    either case, never fabricating a result. A transient failure here is
    NOT distinguished from "no new bar yet" at this layer -- both simply
    mean "cannot resolve on this attempt"; re-running the capture/resolve
    script later will retry naturally."""
    since = date.fromisoformat(since_date_iso)
    end = since + timedelta(days=lookahead_days)
    try:
        with MassiveClient() as client:
            result = client.fetch_daily_bars(ticker, since, end, adjusted=True)
    except (MassiveError, SourceUnavailableError):
        return []  # provider unavailable/entitlement/rate-limit issue -- controlled empty result, not a crash
    bars = []
    for bar in result.bars:
        bar_date = bar.t.date().isoformat()
        if bar_date > since_date_iso:
            bars.append(LiveBar(trading_date=bar_date, close=bar.c))
    return sorted(bars, key=lambda b: b.trading_date)
