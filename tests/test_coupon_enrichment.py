import math
from pathlib import Path

import pandas as pd

from coupon_enrichment import add_streck_volatility_column, enrich_coupon_with_history_form
from europatipset import (
    _odd_mv_triplet,
    _streck_mv_triplet,
    adjust_probs_form_and_volatility,
    adjust_probs_manual_context,
)


def _tf(v, default=float("nan")):
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return default


def test_odd_mv_triplet_relative_change():
    o1, ox, o2 = _odd_mv_triplet([2.0, 3.0, 4.0], [2.0, 3.0, 5.0], _tf)
    assert abs(o1) < 1e-9
    assert abs(ox) < 1e-9
    assert abs(o2 - (4.0 - 5.0) / 5.0) < 1e-9


def test_streck_mv_triplet_percentage_points_to_fraction():
    s1, sx, s2 = _streck_mv_triplet([40.0, 30.0, 30.0], [35.0, 35.0, 30.0], _tf)
    assert abs(s1 - 0.05) < 1e-9
    assert abs(sx - (-0.05)) < 1e-9
    assert abs(s2) < 1e-9


def test_add_streck_volatility_column_sums_abs():
    df = pd.DataFrame(
        {
            "StreckMv1": [0.02, float("nan")],
            "StreckMvX": [-0.01, 0.03],
            "StreckMv2": [0.01, 0.01],
        }
    )
    out = add_streck_volatility_column(df)
    assert abs(out["StreckVol"].iloc[0] - 0.04) < 1e-9
    assert abs(out["StreckVol"].iloc[1] - 0.04) < 1e-9


def test_enrich_coupon_with_history_form_points(tmp_path: Path) -> None:
    data = tmp_path / "data"
    raw = data / "raw"
    inp = data / "input"
    raw.mkdir(parents=True)
    inp.mkdir(parents=True)
    hist = pd.DataFrame(
        {
            "HomeTeam": ["Alpha FC"],
            "AwayTeam": ["Beta United"],
            "Date": ["2024-06-01"],
            "FTR": ["H"],
        }
    )
    hist.to_csv(raw / "history.csv", index=False)
    coupon_path = inp / "coupon.csv"
    coupon_df = pd.DataFrame(
        {
            "Match": ["Alpha FC - Beta United"],
            "Odd1": [2.0],
            "OddX": [3.0],
            "Odd2": [3.5],
            "Streck1": [0.4],
            "StreckX": [0.3],
            "Streck2": [0.3],
        }
    )
    out = enrich_coupon_with_history_form(coupon_df, coupon_path)
    assert out["FormH_pts5"].iloc[0] == 3.0
    assert out["FormB_pts5"].iloc[0] == 0.0


def test_adjust_form_shifts_mass_toward_better_form():
    p_home = adjust_probs_form_and_volatility(0.34, 0.33, 0.33, 15.0, 0.0, float("nan"), form_strength=0.05)
    p_away = adjust_probs_form_and_volatility(0.34, 0.33, 0.33, 0.0, 15.0, float("nan"), form_strength=0.05)
    assert p_home[0] > p_away[0]
    assert p_home[2] < p_away[2]


def test_adjust_volatility_moves_toward_uniform():
    ent = lambda t: float(-sum(x * math.log(x + 1e-12) for x in t))
    base = (0.55, 0.28, 0.17)
    flat = adjust_probs_form_and_volatility(*base, float("nan"), float("nan"), 0.2)
    assert ent(flat) > ent(base) - 1e-6


def test_adjust_form_disabled_ignores_form_diff():
    a = adjust_probs_form_and_volatility(0.34, 0.33, 0.33, 15.0, 0.0, float("nan"), enable_form=False)
    b = adjust_probs_form_and_volatility(0.34, 0.33, 0.33, 0.0, 15.0, float("nan"), enable_form=False)
    assert abs(a[0] - b[0]) < 1e-9
    assert abs(a[2] - b[2]) < 1e-9


def test_adjust_manual_context_biases_selected_outcome():
    base = (0.40, 0.30, 0.30)
    home_up = adjust_probs_manual_context(*base, 1.0, 0.0, -1.0, strength=0.10)
    assert home_up[0] > base[0]
    assert home_up[2] < base[2]
