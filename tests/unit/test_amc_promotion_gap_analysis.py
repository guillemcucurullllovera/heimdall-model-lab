"""Trip-wire regression tests for `reports/MODEL_LAB_AMC_PROMOTION_GAP_
ANALYSIS.md`'s own factual claims. If any of these ever fail, that
document is stale and must be re-audited and updated -- these tests exist
so that drift is caught mechanically rather than silently invalidating a
written audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "model_lab"
REPORTS_DIR = REPO_ROOT / "reports"

PNL_TERMS = re.compile(
    r"\bportfolio\b|\bp&l\b|\bpnl\b|position_siz|sharpe_ratio|max_drawdown|transaction_cost|"
    r"\bturnover\b|backtest_pnl|decision_value|risk_budget",
    re.IGNORECASE,
)
KNOWN_PNL_TERM_FILES = {
    "eval/scores.py", "eval/var_es_backtests.py", "gate_v2/branch_f.py",
    # Added when the AMC Promotion Evaluation framework was built (reports/MODEL_LAB_AMC_
    # PROMOTION_EVALUATION.md): these two files define the SCHEMA for an economic evaluation
    # (field names like `transaction_costs`/`turnover`/`sharpe_ratio`) but every such field is
    # `None`/empty for all 26 models -- no real P&L/portfolio computation exists in either file.
    # Re-verified at the time this set was last extended (2026-10-01); re-audit before trusting
    # this comment if either file's actual logic changes.
    "contracts/amc_promotion_evaluation.py", "command_center_export/amc_promotion_evaluator.py",
    # Added when the Model Lab Empirical Evaluation / Research Validation Layer was built
    # (reports/MODEL_LAB_EMPIRICAL_EVALUATION.md): this IS real P&L/portfolio-style computation
    # this time -- a real volatility-targeting backtest (sharpe/drawdown/turnover/transaction
    # costs, Moreira & Muir 2017) on six real ETFs. It remains Model-Lab-internal research: it
    # never touches AMC, never claims AMC eligibility (`amc_eligible.value` is always False,
    # enforced by test_empirical_evaluator_export.py::test_amc_eligible_is_false_for_every_
    # single_evaluated_model), and does not change any conclusion in the gap analysis document
    # (still: no model has earned AMC promotion). It is the economic-evaluation layer that
    # document identified as missing -- not the AMC integration itself.
    "contracts/economic_evaluation.py", "command_center_export/empirical_evaluator.py",
    "economic_evaluation/__init__.py", "economic_evaluation/vol_targeting.py",
    "economic_evaluation/tail_capital_efficiency.py", "economic_evaluation/risk_reduction_utility.py",
}


def test_no_new_pnl_or_portfolio_simulation_code_has_appeared() -> None:
    """If this ever fails because a NEW file matched, that is not
    necessarily bad -- it may mean someone has started building the
    economic/P&L evaluation this audit found missing. Either way, the gap
    analysis document must be re-read and updated before this test's
    known-file set is simply extended to match."""
    hits: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PNL_TERMS.search(text):
            hits.add(str(path.relative_to(SRC_ROOT)))
    assert hits == KNOWN_PNL_TERM_FILES, (
        f"P&L/portfolio-related terms found in unexpected file(s): {hits - KNOWN_PNL_TERM_FILES} "
        f"-- re-audit reports/MODEL_LAB_AMC_PROMOTION_GAP_ANALYSIS.md before updating this test"
    )


def test_gate_v2_has_never_been_invoked_against_a_real_candidate() -> None:
    """No script or report may construct a real GateResult or call
    Gate v2's own evaluate_branch_* functions outside the gate_v2 package
    itself -- the moment one does, this audit's §5 finding is stale.

    Regression note (2026-10-01): the original `GateResult\\(` pattern (no
    word boundary) false-matched Model Lab's OWN unrelated
    `PromotionGateResult(` class (a different, Model-Lab-side contract,
    not Gate v2's real `GateResult`) as soon as it was introduced -- fixed
    with `\\b` so only a bare `GateResult(` call trips this."""
    pattern = re.compile(r"evaluate_branch_s\(|evaluate_branch_f\(|evaluate_branch_p\(|\bGateResult\(")
    for root in (SRC_ROOT, REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or "gate_v2" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert not pattern.search(text), f"{path}: references Gate v2's evaluation machinery outside gate_v2/ -- re-audit §5"
    for path in REPORTS_DIR.rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert '"GateResult"' not in text and '"gate_verdict"' not in text, f"{path}: appears to contain a real GateResult -- re-audit §5"


def test_model2_prospective_evidence_matches_the_audited_six_and_six() -> None:
    """Pins the exact counts this audit's §2 reports. If MODEL-2's own
    prospective pipeline is re-run and these numbers change, §2 must be
    re-read against the new files and updated, not silently left stale."""
    prospective_dir = REPORTS_DIR / "prospective"
    resolved = sorted(prospective_dir.glob("resolved_*.json"))
    pending = sorted(prospective_dir.glob("pending_*.json"))
    assert len(resolved) == 6, f"expected 6 resolved observations, found {len(resolved)} -- re-audit §2"
    assert len(pending) == 12, f"expected 12 pending files (6 resolved-and-also-pending-at-capture + 6 still-pending), found {len(pending)} -- re-audit §2"
    for path in resolved:
        d = json.loads(path.read_text())
        assert d["is_historical_replay"] is False, f"{path}: now a replay? -- re-audit §2"

    still_pending_tickers = set()
    for path in pending:
        d = json.loads(path.read_text())
        resolved_match = prospective_dir / f"resolved_{d['ticker']}_{d['information_cutoff_date']}.json"
        if not resolved_match.exists():
            still_pending_tickers.add(d["ticker"])
    assert still_pending_tickers == {"SPY", "QQQ", "TLT", "IEF", "GLD", "USO"}, (
        f"expected all 6 tickers to have exactly one still-genuinely-pending observation, got {still_pending_tickers} -- re-audit §2"
    )


def test_candidate_resolved_error_always_equals_exactly_one_constituent() -> None:
    """Pins the structural fact §2 states: a 3-way median's own value is
    always exactly one of its 3 constituents."""
    for path in sorted((REPORTS_DIR / "prospective").glob("resolved_*.json")):
        d = json.loads(path.read_text())
        candidate = d["candidate_forecast_error"]
        benchmarks = d["benchmark_forecast_errors"]
        matches = [m for m in ("EWMA_0.94", "GARCH_t", "EGARCH_t") if benchmarks[m] == candidate]
        assert matches, f"{path}: candidate_forecast_error {candidate!r} matches none of its 3 constituents"
