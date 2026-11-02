#!/usr/bin/env python3
"""MODEL-2F: produces ONE dated shadow snapshot (real `ModelVersion` +
`ModelOutput` records, `lifecycle_state=SHADOW`, `decision_relevant=False`)
for T-VOL's GARCH_t, EWMA_0.94, EGARCH_t, and Combo_median_EWMA_GARCH_EGARCH,
plus T-TAIL's GJR_t_filtered_GPD_FHS, across the six-ETF universe, using the
LATEST available day in each real series.

Remediation (remediation-specification item 4 / independent review P1-D):
extended from GARCH_t/tail-only to all four current T-VOL vol-ladder
challengers -- EWMA_0.94, EGARCH_t, and the median combination previously
had no shadow/ModelOutput identity at all.

Remediation (2026-10-02 Phase 2, independent review finding P2-5): further
extended to one representative challenger per remaining MODEL-2B-2D family
-- the 2-state regime blend (T-STATE, per-ticker), BOCPD (T-CHANGE,
per-ticker), and a Ledoit-Wolf Mahalanobis anomaly scorer (T-ANOM,
panel-level -- see `shadow/build.py::build_anomaly_mahalanobis_shadow_
record`'s own docstring for why this one record is NOT per-ticker).

Run manually, once. NOT scheduled -- see package docstring
(`src/model_lab/shadow/__init__.py`) for the explicit list of what this
does NOT do."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.anomaly.panel import build_return_panel
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.shadow.build import (
    build_anomaly_mahalanobis_shadow_record,
    build_bocpd_changepoint_shadow_record,
    build_combo_vol_shadow_record,
    build_egarch_t_vol_shadow_record,
    build_ewma_vol_shadow_record,
    build_regime_2state_shadow_record,
    build_tail_shadow_records,
    build_vol_shadow_records,
)
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.data import load_return_series

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "shadow"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = DatasetSnapshotRegistry.load_from_directory()
    snapshot_at = datetime.now(UTC).isoformat()
    records: dict[str, dict] = {}

    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        dates = series.dates[1:]
        train_ends, _ = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        entry: dict = {"as_of_date": dates[-1]}
        try:
            vol_version, vol_output = build_vol_shadow_records(series, fold_train_ends=train_ends)
            entry["vol"] = {
                "model_version": vol_version.model_dump(mode="json"),
                "model_output": vol_output.model_dump(mode="json"),
            }
            print(f"{ticker} T-VOL shadow: point_estimate={vol_output.point_estimate}")
        except ValueError as e:
            entry["vol"] = {"status": "SKIPPED", "reason": str(e)}
            print(f"{ticker} T-VOL shadow: SKIPPED ({e})")
        try:
            tail_version, tail_output = build_tail_shadow_records(series, fold_train_ends=train_ends)
            entry["tail"] = {"model_version": tail_version.model_dump(mode="json"), "model_output": tail_output.model_dump(mode="json")}
            print(f"{ticker} T-TAIL shadow: quantiles={[q.value for q in tail_output.quantiles]}")
        except ValueError as e:
            entry["tail"] = {"status": "SKIPPED", "reason": str(e)}
            print(f"{ticker} T-TAIL shadow: SKIPPED ({e})")

        # Remediation (remediation-specification item 4 / independent review
        # P1-D): EWMA_0.94, EGARCH_t, and the median combination previously
        # had no shadow record at all -- only GARCH_t (T-VOL) and the tail
        # model (T-TAIL) did.
        for key, label, builder, kwargs in [
            ("ewma", "EWMA_0.94", build_ewma_vol_shadow_record, {}),
            ("egarch_t", "EGARCH_t", build_egarch_t_vol_shadow_record, {"fold_train_ends": train_ends}),
            ("combo_median", "Combo_median_EWMA_GARCH_EGARCH", build_combo_vol_shadow_record, {"fold_train_ends": train_ends}),
        ]:
            try:
                version, output = builder(series, **kwargs)
                entry[key] = {"model_version": version.model_dump(mode="json"), "model_output": output.model_dump(mode="json")}
                print(f"{ticker} T-VOL/{label} shadow: point_estimate={output.point_estimate}")
            except ValueError as e:
                entry[key] = {"status": "SKIPPED", "reason": str(e)}
                print(f"{ticker} T-VOL/{label} shadow: SKIPPED ({e})")

        # Remediation (2026-10-02 Phase 2 / independent review P2-5):
        # regime (T-STATE) and BOCPD (T-CHANGE) previously had no shadow
        # record at all -- per-ticker, same as every T-VOL/T-TAIL builder.
        try:
            regime_version, regime_output = build_regime_2state_shadow_record(series, fold_train_ends=train_ends)
            entry["regime_2state"] = {
                "model_version": regime_version.model_dump(mode="json"),
                "model_output": regime_output.model_dump(mode="json"),
            }
            print(f"{ticker} T-STATE/regime_2state shadow: point_estimate={regime_output.point_estimate}")
        except ValueError as e:
            entry["regime_2state"] = {"status": "SKIPPED", "reason": str(e)}
            print(f"{ticker} T-STATE/regime_2state shadow: SKIPPED ({e})")
        try:
            bocpd_version, bocpd_output = build_bocpd_changepoint_shadow_record(series)
            entry["changepoint_bocpd"] = {
                "model_version": bocpd_version.model_dump(mode="json"),
                "model_output": bocpd_output.model_dump(mode="json"),
            }
            print(f"{ticker} T-CHANGE/changepoint_bocpd shadow: point_estimate={bocpd_output.point_estimate}")
        except ValueError as e:
            entry["changepoint_bocpd"] = {"status": "SKIPPED", "reason": str(e)}
            print(f"{ticker} T-CHANGE/changepoint_bocpd shadow: SKIPPED ({e})")

        records[ticker] = entry

    # Remediation (2026-10-02 Phase 2 / independent review P2-5): T-ANOM's
    # Mahalanobis scorer is panel-level, not per-ticker (see `shadow/
    # build.py`'s own docstring) -- one record, stored outside the
    # per-ticker `records` dict.
    panel_entry: dict = {}
    try:
        panel = build_return_panel(SIX_ETFS, registry)
        panel_train_ends, _ = expanding_fold_train_ends(panel.dates, horizon=1, min_train_groups=2)
        anomaly_version, anomaly_output = build_anomaly_mahalanobis_shadow_record(
            panel, fold_train_ends=panel_train_ends, estimator="ledoit_wolf"
        )
        panel_entry["anomaly_ledoit_wolf"] = {
            "as_of_date": panel.dates[-1],
            "model_version": anomaly_version.model_dump(mode="json"),
            "model_output": anomaly_output.model_dump(mode="json"),
        }
        print(f"panel T-ANOM/anomaly_ledoit_wolf shadow: point_estimate={anomaly_output.point_estimate}")
    except ValueError as e:
        panel_entry["anomaly_ledoit_wolf"] = {"status": "SKIPPED", "reason": str(e)}
        print(f"panel T-ANOM/anomaly_ledoit_wolf shadow: SKIPPED ({e})")

    out_path = OUT_DIR / f"snapshot_{snapshot_at.replace(':', '-')}.json"
    out_path.write_text(
        json.dumps(
            {"snapshot_at_utc": snapshot_at, "records": records, "panel_records": panel_entry},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
