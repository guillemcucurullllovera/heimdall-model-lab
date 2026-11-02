#!/usr/bin/env python3
"""Remediation of independent-review finding P2-2: applies Holm-Bonferroni
(FWER) and Benjamini-Hochberg (FDR) correction to the two 24-cell DM-test
families reported throughout MODEL2A/MODEL2G -- "best single model beats
naive" and "combination beats GARCH_t" -- which were previously tallied at
the uncorrected alpha=0.05 threshold with no treatment of the fact that each
is a family of 24 independent tests (one per ticker x horizon cell).

Reads ONLY the already-committed `reports/vol_ladder/*.json` (no re-fit, no
re-run of any model); this is a pure post-hoc statistical correction over
already-computed p-values. Writes `reports/vol_ladder/_multiple_testing_correction.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lab.eval.multiple_testing import benjamini_hochberg, holm_bonferroni
from model_lab.tasks.registry import SIX_ETFS

HORIZONS = [1, 5, 21, 45]
LADDER_DIR = Path(__file__).resolve().parents[1] / "reports" / "vol_ladder"
OUT_PATH = LADDER_DIR / "_multiple_testing_correction.json"


def main() -> None:
    family1_cells = []  # best non-naive model vs naive
    family2_cells = []  # combination vs GARCH_t

    for ticker in SIX_ETFS:
        for horizon in HORIZONS:
            d = json.loads((LADDER_DIR / f"{ticker}_h{horizon}.json").read_text())
            results = d["results"]
            non_naive = {m: v for m, v in results.items() if m != "naive_trailing_rv"}
            best_model = min(non_naive, key=lambda m: non_naive[m]["qlike_mean"])
            p1 = non_naive[best_model]["dm_vs_naive_p"]
            family1_cells.append({"ticker": ticker, "horizon": horizon, "best_model": best_model, "dm_vs_naive_p": p1})

            p2 = results["Combo_median_EWMA_GARCH_EGARCH"]["dm_vs_garch_t_p"]
            family2_cells.append({"ticker": ticker, "horizon": horizon, "dm_vs_garch_t_p": p2})

    p1_values = [c["dm_vs_naive_p"] for c in family1_cells]
    holm1 = holm_bonferroni(p1_values, alpha=0.05)
    bh1 = benjamini_hochberg(p1_values, alpha=0.05)
    for c, h, b in zip(family1_cells, holm1, bh1, strict=True):
        c["uncorrected_significant_at_0.05"] = c["dm_vs_naive_p"] < 0.05
        c["holm_bonferroni_significant"] = bool(h)
        c["benjamini_hochberg_significant"] = bool(b)

    p2_values = [c["dm_vs_garch_t_p"] for c in family2_cells]
    holm2 = holm_bonferroni(p2_values, alpha=0.05)
    bh2 = benjamini_hochberg(p2_values, alpha=0.05)
    for c, h, b in zip(family2_cells, holm2, bh2, strict=True):
        c["uncorrected_significant_at_0.05"] = c["dm_vs_garch_t_p"] < 0.05
        c["holm_bonferroni_significant"] = bool(h)
        c["benjamini_hochberg_significant"] = bool(b)

    out = {
        "method": "Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR), each applied "
        "independently within its own family of m=24 (ticker, horizon) cells, alpha=0.05",
        "family_1_best_model_vs_naive": {
            "n_cells": 24,
            "uncorrected_count": sum(c["uncorrected_significant_at_0.05"] for c in family1_cells),
            "holm_bonferroni_count": sum(c["holm_bonferroni_significant"] for c in family1_cells),
            "benjamini_hochberg_count": sum(c["benjamini_hochberg_significant"] for c in family1_cells),
            "cells": family1_cells,
        },
        "family_2_combination_vs_garch_t": {
            "n_cells": 24,
            "uncorrected_count": sum(c["uncorrected_significant_at_0.05"] for c in family2_cells),
            "holm_bonferroni_count": sum(c["holm_bonferroni_significant"] for c in family2_cells),
            "benjamini_hochberg_count": sum(c["benjamini_hochberg_significant"] for c in family2_cells),
            "cells": family2_cells,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Family 1 (best vs naive): uncorrected={out['family_1_best_model_vs_naive']['uncorrected_count']}/24, "
          f"Holm={out['family_1_best_model_vs_naive']['holm_bonferroni_count']}/24, "
          f"BH={out['family_1_best_model_vs_naive']['benjamini_hochberg_count']}/24")
    print(f"Family 2 (combo vs GARCH_t): uncorrected={out['family_2_combination_vs_garch_t']['uncorrected_count']}/24, "
          f"Holm={out['family_2_combination_vs_garch_t']['holm_bonferroni_count']}/24, "
          f"BH={out['family_2_combination_vs_garch_t']['benjamini_hochberg_count']}/24")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
