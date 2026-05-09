import pandas as pd

from free_context import _parse_standings_json, _pick_competition_for_pair, split_match_cell


def test_split_match_cell():
    assert split_match_cell("Arsenal FC - West Ham United") == ("Arsenal FC", "West Ham United")


def test_parse_standings_total():
    payload = {
        "standings": [
            {
                "type": "TOTAL",
                "table": [
                    {
                        "position": 1,
                        "team": {"name": "Arsenal FC"},
                        "playedGames": 10,
                        "won": 8,
                        "draw": 1,
                        "lost": 1,
                        "points": 25,
                        "goalsFor": 22,
                        "goalsAgainst": 9,
                        "goalDifference": 13,
                    },
                    {
                        "position": 15,
                        "team": {"name": "West Ham United FC"},
                        "playedGames": 10,
                        "won": 2,
                        "draw": 3,
                        "lost": 5,
                        "points": 9,
                        "goalsFor": 10,
                        "goalsAgainst": 18,
                        "goalDifference": -8,
                    },
                ],
            }
        ]
    }
    roster = _parse_standings_json(payload)
    assert "arsenal fc" in roster
    assert roster["arsenal fc"]["points"] == 25


def test_pick_competition_for_pair_fuzzy():
    standings_by_comp = {
        "PL": _parse_standings_json(
            {
                "standings": [
                    {
                        "type": "TOTAL",
                        "table": [
                            {
                                "position": 1,
                                "team": {"name": "Arsenal FC"},
                                "playedGames": 5,
                                "won": 5,
                                "draw": 0,
                                "lost": 0,
                                "points": 15,
                                "goalsFor": 10,
                                "goalsAgainst": 2,
                                "goalDifference": 8,
                            },
                            {
                                "position": 10,
                                "team": {"name": "West Ham United FC"},
                                "playedGames": 5,
                                "won": 1,
                                "draw": 1,
                                "lost": 3,
                                "points": 4,
                                "goalsFor": 5,
                                "goalsAgainst": 9,
                                "goalDifference": -4,
                            },
                        ],
                    }
                ]
            }
        )
    }
    code, hi, ai, conf = _pick_competition_for_pair("Arsenal", "West Ham", standings_by_comp)
    assert code == "PL"
    assert hi["points"] == 15
    assert ai["points"] == 4
    assert conf > 0.7


def test_form_last_n_from_history(tmp_path):
    from free_context import _form_last_n

    p = tmp_path / "h.csv"
    p.write_text(
        "Date,HomeTeam,AwayTeam,FTR,Competition\n"
        "2026-05-01,Arsenal FC,Brentford FC,H,PL\n"
        "2026-05-03,Chelsea FC,Arsenal FC,A,PL\n",
        encoding="utf-8",
    )
    df = pd.read_csv(p)
    f = _form_last_n("Arsenal FC", "PL", df, n=5)
    assert f.startswith("W") or f.startswith("L")
