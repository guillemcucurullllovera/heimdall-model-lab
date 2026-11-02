#!/usr/bin/env python3
"""Phase 5 (MODEL-2 non-review continuation): capture + (attempt) resolve
one round of genuine prospective evidence for the MODEL-2G candidate,
across the six-ETF universe.

Run manually, once per invocation -- NOT scheduled (no cron/service is
created or activated by this script; see `prospective/__init__.py`'s own
docstring for the full list of what this deliberately does not do). Safe
to re-run: capture is idempotent (skips a ticker that already has a pending
observation at the CURRENT information cutoff date), and resolve only
writes a new resolved record for an observation that is still pending and
for which the live check actually found new data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.prospective.capture import capture_pending_observation, pending_observation_path
from model_lab.prospective.live_extend import extend_series_with_live_bars
from model_lab.prospective.records import PendingObservation
from model_lab.prospective.resolve import resolve_pending_observation
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

STORE_DIR = Path(__file__).resolve().parents[1] / "reports" / "prospective"


def main() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()

    print("=== CAPTURE ===")
    for ticker in SIX_ETFS:
        base_series = load_return_series(ticker, registry)
        # Roll the information cutoff forward as far as REAL, live-fetched data allows (never
        # fabricated, never reading from the frozen registry beyond what it actually has) -- this
        # is what lets a repeatedly re-invoked (never automated) runner advance day-over-day.
        extended = extend_series_with_live_bars(base_series)
        series = extended if extended is not None else base_series
        cutoff = series.dates[-1]
        path = pending_observation_path(STORE_DIR, ticker, cutoff)
        if path.exists():
            print(f"{ticker}: already captured at cutoff={cutoff} ({path.name}) -- skipping")
            continue
        pending = capture_pending_observation(series)
        path.write_text(pending.model_dump_json(indent=2), encoding="utf-8")
        extended_note = " (live-extended beyond the frozen registry)" if extended is not None else ""
        print(
            f"{ticker}: captured, cutoff={cutoff}{extended_note}, "
            f"candidate_point_estimate={pending.candidate_output.point_estimate}, hash={pending.pending_observation_hash[:12]}..."
        )
        time.sleep(5)  # stay well under the provider's rate limit (extend_series_with_live_bars calls it too)

    print("\n=== RESOLVE ===")
    for pending_path in sorted(STORE_DIR.glob("pending_*.json")):
        pending = PendingObservation.model_validate_json(pending_path.read_text())
        resolved_path = STORE_DIR / f"resolved_{pending.ticker}_{pending.information_cutoff_date}.json"
        if resolved_path.exists():
            print(f"{pending.ticker} (cutoff={pending.information_cutoff_date}): already resolved -- skipping")
            continue
        time.sleep(5)  # stay well under the provider's rate limit across sequential tickers
        resolved = resolve_pending_observation(pending)
        if resolved is None:
            print(f"{pending.ticker} (cutoff={pending.information_cutoff_date}): still PENDING -- no new real-world data yet")
            continue
        resolved_path.write_text(resolved.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"{pending.ticker} (cutoff={pending.information_cutoff_date}): RESOLVED against "
            f"{resolved.realized_target_date} -- candidate_qlike_loss={resolved.candidate_forecast_error}, "
            f"is_historical_replay={resolved.is_historical_replay}"
        )

    summary = {
        "n_pending_files": len(list(STORE_DIR.glob("pending_*.json"))),
        "n_resolved_files": len(list(STORE_DIR.glob("resolved_*.json"))),
    }
    (STORE_DIR / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
