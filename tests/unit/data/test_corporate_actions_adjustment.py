"""Unit tests for P0-4 remediation: the split-adjustment direction, the
real XNYS calendar gap detection, and (regression) that the derived
split-adjusted USO series has no corporate-action discontinuity.

MODEL1_ADVERSARIAL_REVIEW.md finding C.4: USO's raw 2020-04-29 1-for-8
reverse split produces a mechanical +213% single-day return, invisible to
the mechanical gate. These tests pin the FIXED behaviour (and, via
`test_price_adjustment_factor_is_reciprocal_of_share_count_ratio`, the
specific sign-error class this remediation itself hit and corrected while
building the fix -- see providers/massive.py's SplitEvent docstring).
"""

from __future__ import annotations

import json
import math
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from model_lab.data.providers.massive import SplitEvent
from model_lab.data.snapshot import compute_calendar_gaps

USO_ADJUSTED_ROWS = Path("./local-data/snapshots/uso_daily_split_adjusted/rows.ndjson")
SPY_RAW_ROWS = Path("./local-data/snapshots/alpaca_SPY_daily_raw/rows.ndjson")


def test_price_adjustment_factor_is_reciprocal_of_share_count_ratio() -> None:
    # USO's real 1-for-8 reverse split, as returned by Massive/Polygon.
    ev = SplitEvent(ticker="USO", execution_date=date(2020, 4, 29), split_from="8", split_to="1")
    price_factor = Decimal(ev.price_adjustment_factor_pre_split())
    share_count_ratio = Decimal(ev.share_count_ratio_new_per_old())
    assert price_factor == Decimal(8)
    assert share_count_ratio == Decimal("0.125")
    assert price_factor * share_count_ratio == 1  # exact reciprocals, by construction
    # The defect this remediation fixed: using share_count_ratio (0.125) to
    # multiply PRICES would move a pre-split price DOWN, worsening (not
    # fixing) the discontinuity. price_factor (8) is the correct direction.
    pre_split_close = Decimal("2.13")
    assert pre_split_close * price_factor == Decimal("17.04")
    assert pre_split_close * share_count_ratio == Decimal("0.26625")  # the wrong-direction result, pinned as a negative example


@pytest.mark.skipif(not USO_ADJUSTED_ROWS.exists(), reason="derived USO snapshot not present on this host")
def test_derived_uso_series_has_no_split_discontinuity() -> None:
    rows = [json.loads(line) for line in USO_ADJUSTED_ROWS.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["observation_period"])
    prev_close = None
    max_abs_log_return = 0.0
    for r in rows:
        bar = json.loads(r["value"])
        close = float(bar["c"])
        if prev_close is not None and prev_close > 0 and close > 0:
            log_ret = abs(math.log(close / prev_close))
            max_abs_log_return = max(max_abs_log_return, log_ret)
        prev_close = close
    # A genuine +213% single-day artifact would be log(1+2.13)=~1.14; assert
    # nothing remotely close to that survives in the adjusted series (real
    # single-day ETF moves are well under 50%, log(1.5)=~0.41).
    assert max_abs_log_return < math.log(1.5), f"unexpected large single-day move survived adjustment: {max_abs_log_return}"


@pytest.mark.skipif(not USO_ADJUSTED_ROWS.exists(), reason="derived USO snapshot not present on this host")
def test_derived_uso_manifest_names_its_lineage() -> None:
    manifest = json.loads(
        (USO_ADJUSTED_ROWS.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["adjustments"]["kind"] == "split_adjusted"
    assert manifest["adjustments"]["corporate_actions_snapshot_id"] == "corporate_actions_USO"
    assert manifest["adjustments"]["method"] is not None
    assert "alpaca_USO_daily_raw" in manifest["params"]["raw_snapshot_id"]


def test_raw_uso_snapshot_was_never_modified_in_place() -> None:
    """The raw snapshot must still show the UNADJUSTED discontinuity -- the
    remediation must never mutate raw vendor data in place (§9 rule 3)."""
    raw_path = Path("./local-data/snapshots/alpaca_USO_daily_raw/rows.ndjson")
    if not raw_path.exists():
        pytest.skip("raw USO snapshot not present on this host")
    rows = {json.loads(line)["observation_period"]: json.loads(json.loads(line)["value"]) for line in raw_path.read_text(encoding="utf-8").splitlines()}
    assert rows["2020-04-28"]["c"] == "2.13"
    assert rows["2020-04-29"]["o"] == "18.01"  # still raw/unadjusted, exactly as originally acquired


@pytest.mark.skipif(not SPY_RAW_ROWS.exists(), reason="SPY snapshot not present on this host")
def test_real_calendar_gap_is_zero_for_spy() -> None:
    rows = [json.loads(line) for line in SPY_RAW_ROWS.read_text(encoding="utf-8").splitlines()]
    dates = [date.fromisoformat(r["observation_period"]) for r in rows]
    gaps = compute_calendar_gaps(dates)
    assert gaps == [], f"expected 0 true XNYS calendar gaps, found {gaps}"


def test_calendar_gap_detects_a_real_injected_gap() -> None:
    """Sanity check that compute_calendar_gaps is not vacuously always-empty:
    remove a known trading day and confirm it's detected."""
    base = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]  # all real XNYS trading days
    with_gap = [base[0], base[1], base[3]]  # drop 2024-01-04
    gaps = compute_calendar_gaps(with_gap)
    assert len(gaps) == 1
    assert gaps[0].start == date(2024, 1, 4)
    assert gaps[0].end == date(2024, 1, 4)
