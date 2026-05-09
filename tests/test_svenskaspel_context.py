import pytest

from svenskaspel_context import build_round_context_bundle, context_matches_dataframe


def test_build_round_context_bundle_minimal():
    state = {
        "Draws": {
            "ids": ["2_999"],
            "entities": {
                "2_999": {
                    "drawNumber": 999,
                    "drawComment": "Test",
                    "drawState": "Open",
                    "drawEvents": [
                        {"eventNumber": 1, "eventDescription": "A - B", "eventComment": "Nyhet X"},
                    ],
                }
            },
        },
        "BetEvents": {
            "2_999_1": {
                "matchId": 111,
                "eventDescription": "A - B",
                "eventNumber": 1,
                "leagueId": 5,
                "participants": [1, 2],
                "externalId": {"sportradarId": "sr1", "Kambi": "k1"},
                "eventTypeStatisticId": "111_1_1",
            }
        },
        "Participants": {
            "1": {"id": 1, "type": "home", "name": "Alpha"},
            "2": {"id": 2, "type": "away", "name": "Beta"},
        },
        "Leagues": {"5": {"id": 5, "name": "Testliga", "countryId": 9}},
        "Countries": {"9": {"id": 9, "name": "Testland"}},
        "SportEvents": {
            "111": {
                "matchId": 111,
                "matchStart": "2026-01-01T15:00:00+01:00",
                "sportEventStatus": "NotStarted",
            }
        },
    }
    bundle = build_round_context_bundle(state, source_url="https://example.test/")
    assert bundle["draw"]["drawNumber"] == 999
    df = context_matches_dataframe(bundle)
    assert len(df) == 1
    assert df.iloc[0]["homeTeam"] == "Alpha"
    assert df.iloc[0]["eventComment"] == "Nyhet X"
    assert df.iloc[0]["sportradarId"] == "sr1"


def test_fetch_live_round_smoke():
    """Valfritt nätverkstest — körs bara om du sätter RUN_NETWORK_TESTS=1."""
    import os

    if os.getenv("RUN_NETWORK_TESTS") != "1":
        pytest.skip("Sätt RUN_NETWORK_TESTS=1 för live-anrop mot spela.svenskaspel.se.")
    from svenskaspel_context import fetch_europatipset_round_context

    b = fetch_europatipset_round_context()
    assert b.get("matches")
    assert b["matches"][0].get("matchLabel")
