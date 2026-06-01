import numpy as np

from europatipset import (
    coupon_odds_gap_report,
    odds_from_streck_distribution,
    resolve_match_odds,
)


def test_odds_from_streck_distribution():
    o1, ox, o2 = odds_from_streck_distribution(50, 30, 20)
    assert min(o1, ox, o2) > 1.01
    inv = np.array([1 / o1, 1 / ox, 1 / o2])
    assert abs(inv.sum() - 1.0) < 0.15


def test_resolve_match_odds_prefers_current():
    stat = {
        "odds": {"current": {"value": ["2.00", "3.20", "3.80"]}},
        "startOdds": {"current": {"value": ["1.90", "3.10", "4.00"]}},
    }
    o1, ox, o2, src = resolve_match_odds(stat, (40, 30, 30))
    assert src == "current"
    assert o1 == 2.0


def test_resolve_match_odds_streck_fallback():
    stat = {
        "odds": {"current": {"value": [None, None, None]}},
        "startOdds": {"current": {"value": [None, None, None]}},
    }
    o1, ox, o2, src = resolve_match_odds(stat, (52, 32, 16))
    assert src == "streck_estimate"
    assert min(o1, ox, o2) > 1.01


def test_coupon_odds_gap_report():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"Match": "A-B", "Odd1": 2.0, "OddX": 3.0, "Odd2": 4.0, "OddsSource": "current"},
            {"Match": "C-D", "Odd1": np.nan, "OddX": np.nan, "Odd2": np.nan, "OddsSource": "missing"},
        ]
    )
    rep = coupon_odds_gap_report(df)
    assert rep["complete"] == 1
    assert len(rep["missing"]) == 1
