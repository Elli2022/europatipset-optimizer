from pathlib import Path

from play_journal import build_settlement_insights, ensure_seed_week_22, learning_hint, load_journal
from round_lessons import (
    analyze_week_22_case,
    column_coverage_count,
    journal_calibration_params,
    spike_risk_score,
)
import numpy as np


def test_week_22_case_counts():
    case = analyze_week_22_case()
    assert case["played_column_coverage"] == 8
    assert case["reference_safe_coverage"] == 9
    assert case["hits_best_row_played"] == 8
    assert len(case["outcomes_13"]) == 13


def test_spike_risk_higher_on_tight_match():
    tight = np.array([0.40, 0.35, 0.25])
    clear = np.array([0.72, 0.18, 0.10])
    assert spike_risk_score(tight) > spike_risk_score(clear)


def test_journal_calibration_tightens_after_misses():
    base = journal_calibration_params(None)
    tight = journal_calibration_params(
        {
            "settled_rounds": 2,
            "miss_on_single_pick": 4,
            "miss_on_single_favorite": 3,
            "column_vs_payout_gap_events": 1,
        }
    )
    assert tight["spike_min_top_prob"] >= base["spike_min_top_prob"]
    assert tight["force_hedge_risk"] <= base["force_hedge_risk"]


def test_build_settlement_insights_payout_gap():
    rows = [{"Match": "A-B", "Förslag": "1X", "P1": 0.5, "PX": 0.3, "P2": 0.2}] * 13
    ins = build_settlement_insights(rows, hits_best_row=8, outcomes_13="1" * 13, payout_threshold=10)
    assert ins["below_payout_threshold"] is True
    assert ins["column_coverage"] == 13


def test_ensure_seed_week_22(tmp_path: Path):
    journal = tmp_path / "play_journal.json"
    created = ensure_seed_week_22(journal, tmp_path)
    assert created is True
    data = load_journal(journal)
    assert data["aggregate"]["settled_rounds"] == 1
    assert "Vecka 22" in data["bets"][0]["draw_comment"]
    assert learning_hint(data)
    assert ensure_seed_week_22(journal, tmp_path) is False


def test_column_coverage_count():
    assert column_coverage_count(["1X", "2", "1"], "121") == 3
