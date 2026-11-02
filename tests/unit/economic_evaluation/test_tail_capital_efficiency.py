from __future__ import annotations

import json
from pathlib import Path

from model_lab.economic_evaluation.tail_capital_efficiency import (
    compute_tail_capital_efficiency,
    summarize_capital_efficiency,
)


def _write_fake_tail_ladder(reports_dir: Path, ticker: str, *, uncond_fz0: dict, cond_fz0: dict) -> None:
    (reports_dir / "tail_ladder").mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            "unconditional_GPD_POT": {
                level: {"fz0_mean": fz0, "christoffersen_lr_cc_p": 0.5} for level, fz0 in uncond_fz0.items()
            },
            "GJR_t_filtered_GPD_FHS": {
                level: {"fz0_mean": fz0, "christoffersen_lr_cc_p": 0.5} for level, fz0 in cond_fz0.items()
            },
        }
    }
    (reports_dir / "tail_ladder" / f"{ticker}.json").write_text(json.dumps(payload))


def test_missing_ladder_file_returns_empty_not_fabricated(tmp_path: Path) -> None:
    assert compute_tail_capital_efficiency(tmp_path, "NOPE") == []


def test_conditional_more_efficient_yields_negative_pct_change(tmp_path: Path) -> None:
    _write_fake_tail_ladder(
        tmp_path, "FAKE",
        uncond_fz0={"0.05": 10.0, "0.025": 10.0, "0.01": 10.0},
        cond_fz0={"0.05": 8.0, "0.025": 8.0, "0.01": 8.0},
    )
    results = compute_tail_capital_efficiency(tmp_path, "FAKE")
    assert len(results) == 3
    for r in results:
        assert float(r.pct_change_conditional_vs_unconditional) == -20.0


def test_conditional_less_efficient_yields_positive_pct_change(tmp_path: Path) -> None:
    _write_fake_tail_ladder(
        tmp_path, "FAKE",
        uncond_fz0={"0.05": 10.0, "0.025": 10.0, "0.01": 10.0},
        cond_fz0={"0.05": 12.0, "0.025": 12.0, "0.01": 12.0},
    )
    results = compute_tail_capital_efficiency(tmp_path, "FAKE")
    for r in results:
        assert float(r.pct_change_conditional_vs_unconditional) == 20.0


def test_summarize_on_empty_list_returns_empty_dict_not_fabricated_summary() -> None:
    assert summarize_capital_efficiency([]) == {}


def test_summarize_counts_cells_where_conditional_is_more_efficient(tmp_path: Path) -> None:
    _write_fake_tail_ladder(
        tmp_path, "FAKE",
        uncond_fz0={"0.05": 10.0, "0.025": 10.0, "0.01": 10.0},
        cond_fz0={"0.05": 8.0, "0.025": 12.0, "0.01": 8.0},
    )
    results = compute_tail_capital_efficiency(tmp_path, "FAKE")
    summary = summarize_capital_efficiency(results)
    assert summary["n_cells"] == 3
    assert summary["n_cells_conditional_more_efficient"] == 2
