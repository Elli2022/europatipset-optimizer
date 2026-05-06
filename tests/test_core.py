import pandas as pd

from europatipset import (
    MatchSuggestion,
    optimize_system,
    recommend_max_stake,
    simulate_rights_distribution,
    validate_coupon_data,
)


def _sample_matches():
    return [
        MatchSuggestion("M1", 2.0, 3.2, 3.6, 0.50, 0.28, 0.22, 0.40, 0.30, 0.30, ""),
        MatchSuggestion("M2", 2.2, 3.1, 3.2, 0.46, 0.29, 0.25, 0.34, 0.33, 0.33, ""),
        MatchSuggestion("M3", 1.8, 3.5, 4.6, 0.56, 0.25, 0.19, 0.50, 0.26, 0.24, ""),
    ]


def test_optimize_system_respects_budget():
    picks, rows = optimize_system(_sample_matches(), max_rows=4, strategy="balanced")
    assert rows <= 4
    assert len(picks) == 3
    assert all(p in {"1", "X", "2", "1X", "12", "X2", "1X2"} for p in picks)


def test_simulate_distribution_has_most_likely():
    result_df = pd.DataFrame(
        {
            "Förslag": ["1", "1X", "1"],
            "P1": [0.55, 0.45, 0.60],
            "PX": [0.25, 0.30, 0.22],
            "P2": [0.20, 0.25, 0.18],
        }
    )
    dist = simulate_rights_distribution(result_df, n_sim=2000, seed=1)
    assert "most_likely" in dist
    assert isinstance(dist["most_likely"], int)


def test_recommend_max_stake_basic():
    forecast = {13: 0.01, 12: 0.03, 11: 0.08, 10: 0.12}
    payouts = {13: 20000, 12: 1200, 11: 120, 10: 30}
    rec = recommend_max_stake(forecast, payouts, margin_pct=10, bankroll_cap=500)
    assert rec["max_break_even"] >= 0
    assert rec["recommended_max"] <= 500


def test_validate_coupon_data_flags_invalid_odds():
    df = pd.DataFrame(
        {
            "Match": ["A-B"],
            "Odd1": [1.0],
            "OddX": [3.0],
            "Odd2": [4.0],
            "Streck1": [90],
            "StreckX": [5],
            "Streck2": [4],
        }
    )
    warnings = validate_coupon_data(df)
    assert any("orimliga odds" in w for w in warnings)
