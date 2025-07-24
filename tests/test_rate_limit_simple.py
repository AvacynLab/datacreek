from datacreek.utils.rate_limit import configure, consume_token


def test_local_token_bucket(monkeypatch):
    configure(rate=2, burst=2)
    assert consume_token("t", now=0)
    assert consume_token("t", now=0)
    assert not consume_token("t", now=0)
    assert consume_token("t", now=1)
