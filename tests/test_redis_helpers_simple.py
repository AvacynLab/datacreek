import pytest
from datacreek.utils import redis_helpers

def test_decode_hash_bytes_and_json():
    data = {b'a': b'1', 'b': b'{"x": 2}'}
    assert redis_helpers.decode_hash(data) == {'a': 1, 'b': {'x': 2}}

def test_decode_hash_error_passthrough(monkeypatch):
    monkeypatch.setattr(redis_helpers.json, 'loads', lambda *_: (_ for _ in ()).throw(ValueError))
    assert redis_helpers.decode_hash({b'bad': b'notjson'}) == {'bad': 'notjson'}
