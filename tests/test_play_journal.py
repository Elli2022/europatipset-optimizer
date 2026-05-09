import json
from pathlib import Path

from play_journal import analyze_column_coverage, settle_bet


def test_analyze_column_coverage_counts_misses():
    rows = [
        {"Match": "A-B", "Förslag": "1", "P1": 0.5, "PX": 0.25, "P2": 0.25},
        {"Match": "C-D", "Förslag": "1X", "P1": 0.4, "PX": 0.35, "P2": 0.25},
    ]
    # Pad to 13 outcomes by repeating pattern (test only needs first 2 rows logic)
    o = "2" + "2" + "1" * 11
    out = analyze_column_coverage(rows, o)
    assert out["covered_columns"] == 0
    assert len(out["misses"]) == 2


def test_settle_updates_aggregate(tmp_path: Path):
    journal = tmp_path / "play_journal.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "bets": [
                    {
                        "id": "bet_test",
                        "status": "pending",
                        "recommendation_rows": [
                            {"Match": "A-B", "Förslag": "1", "P1": 0.55, "PX": 0.25, "P2": 0.2},
                        ]
                        * 13,
                    }
                ],
                "aggregate": {
                    "settled_rounds": 0,
                    "miss_events_total": 0,
                    "miss_on_single_pick": 0,
                    "miss_on_single_favorite": 0,
                    "miss_on_single_underdog": 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outcomes = "X" * 13
    settle_bet(journal, "bet_test", hits_best_row=4, outcomes_13=outcomes)
    data = json.loads(journal.read_text(encoding="utf-8"))
    assert data["bets"][0]["status"] == "settled"
    assert data["aggregate"]["settled_rounds"] == 1
    assert data["aggregate"]["miss_events_total"] == 13
