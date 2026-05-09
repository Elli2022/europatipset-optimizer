from pathlib import Path

import pytest

from europatipset import backtest_strategies, default_signal_backtest_profiles


def test_default_signal_profiles_cover_ablations():
    p = default_signal_backtest_profiles()
    names = {str(x["SignalProfile"]) for x in p}
    assert "allt_på" in names
    assert "utan_form_sannolikhetsjustering" in names
    assert "utan_vol_sannolikhetsjustering" in names


@pytest.mark.parametrize("compare", [False, True])
def test_backtest_smoke_with_bundled_data(compare: bool) -> None:
    repo = Path(__file__).resolve().parents[1]
    hist = repo / "data" / "raw" / "history.csv"
    model = repo / "data" / "models" / "calibration.pkl"
    if not hist.exists() or hist.stat().st_size == 0 or not model.exists():
        pytest.skip("Saknar data/raw/history.csv eller modell för smoke-test.")

    out = backtest_strategies(
        history_csv=hist,
        model_file=model,
        budgets=[32],
        strategies=["balanced"],
        game_type="europatipset",
        n_coupons=4,
        seed=11,
        compare_signal_profiles=compare,
    )
    assert "FullHitRate" in out.columns
    assert (out["ROI"] <= 500).all() and (out["ROI"] >= -1.0).all()
    if compare:
        assert "SignalProfile" in out.columns
        assert len(out) >= len(default_signal_backtest_profiles())
