from pathlib import Path

from play_journal import merge_journal_data


def test_merge_journal_dedupes_by_id(tmp_path: Path):
    a = {
        "version": 1,
        "bets": [
            {"id": "bet_1", "status": "pending", "saved_at": "2026-01-02T00:00:00+00:00"}
        ],
        "aggregate": {"settled_rounds": 0},
    }
    b = {
        "version": 1,
        "bets": [
            {"id": "bet_1", "status": "pending", "saved_at": "2026-01-01T00:00:00+00:00"}
        ],
        "aggregate": {"settled_rounds": 0},
    }
    m = merge_journal_data(a, b)
    assert len(m["bets"]) == 1
    assert m["bets"][0]["saved_at"].startswith("2026-01-02")


def test_merge_prefers_settled_when_duplicate_ids():
    a = {
        "version": 1,
        "bets": [{"id": "bet_1", "status": "pending", "saved_at": "2026-01-03T00:00:00+00:00"}],
        "aggregate": {},
    }
    b = {
        "version": 1,
        "bets": [
            {
                "id": "bet_1",
                "status": "settled",
                "saved_at": "2026-01-02T00:00:00+00:00",
                "settled_at": "2026-01-04T00:00:00+00:00",
                "insights": {"hits_best_row": 10},
            }
        ],
        "aggregate": {},
    }
    m = merge_journal_data(a, b)
    assert m["bets"][0]["status"] == "settled"
