from journal_browser_sync import decode_journal_blob, encode_journal_blob


def test_journal_zlib_roundtrip():
    data = {"version": 1, "bets": [{"id": "x"}], "aggregate": {"settled_rounds": 0}}
    blob = encode_journal_blob(data)
    assert decode_journal_blob(blob) == data
