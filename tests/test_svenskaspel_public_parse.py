from europatipset import extract_correct_row_from_draw, outcome_to_sign, score_text_to_sign


def test_score_text_to_sign_basic():
    assert score_text_to_sign("Slut 2–1") == "1"
    assert score_text_to_sign("1 - 1 text") == "X"


def test_outcome_to_sign_list_winner():
    oc = [{"won": True, "label": "1"}, {"won": False, "label": "X"}]
    assert outcome_to_sign(oc) == "1"


def test_extract_correct_row_from_draw_with_outcomes():
    draw = {
        "drawEvents": [
            {"eventNumber": i + 1, "cancelled": False, "outcomes": [{"won": True, "label": "1"}]}
            for i in range(13)
        ]
    }
    assert extract_correct_row_from_draw(draw) == "1" * 13
