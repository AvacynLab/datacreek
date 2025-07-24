from datacreek.utils.rate_limit import configure, consume_token


def test_local_token_bucket(monkeypatch):
    configure(rate=2, burst=2)
    assert consume_token("t", now=0)
    assert consume_token("t", now=0)
    assert not consume_token("t", now=0)
    assert consume_token("t", now=1)


class FakeRedisModule:
    class exceptions:
        class NoScriptError(Exception):
            pass

    def __init__(self):
        self.loaded = False

    class Redis:
        def __init__(self, parent):
            self.parent = parent
            self.reloads = 0

        def script_load(self, script):
            self.parent.loaded = True
            return "sha"

        def evalsha(self, sha, numkeys, key, rate, burst, now):
            return 1


class NoScriptRedis(FakeRedisModule):
    class Redis(FakeRedisModule.Redis):
        def __init__(self, parent):
            super().__init__(parent)
            self.called = False

        def evalsha(self, sha, numkeys, key, rate, burst, now):
            if not self.called:
                self.called = True
                raise NoScriptRedis.exceptions.NoScriptError()
            return 1


class ErrorRedis(FakeRedisModule):
    class Redis(FakeRedisModule.Redis):
        def evalsha(self, sha, numkeys, key, rate, burst, now):
            raise Exception("boom")


def test_redis_token_bucket(monkeypatch):
    fake = FakeRedisModule()
    client = fake.Redis(fake)
    monkeypatch.setattr("datacreek.utils.rate_limit.redis", fake)
    configure(client=client, rate=1, burst=1)
    assert consume_token("u", now=0, client=client)
    assert client.parent.loaded


def test_redis_fallback_to_local(monkeypatch):
    fake = ErrorRedis()
    client = fake.Redis(fake)
    monkeypatch.setattr("datacreek.utils.rate_limit.redis", fake)
    configure(client=client, rate=1, burst=1)
    assert consume_token("x", now=0, client=client)
    assert not consume_token("x", now=0, client=client)
