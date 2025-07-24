import importlib
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, "./")

fake_pydantic = types.ModuleType("pydantic")


class _BaseModel:
    model_config = None

    def model_dump(self):
        return {}


def _config_dict(**kw):
    return kw


def _field(default=None, **_kw):
    return default


fake_pydantic.BaseModel = _BaseModel
fake_pydantic.ConfigDict = _config_dict
fake_pydantic.Field = _field
fake_pydantic.ValidationError = Exception
sys.modules.setdefault("pydantic", fake_pydantic)


def setup_module(module):
    # Ensure fresh import without background thread and heavy deps
    cfg_stub = types.ModuleType("datacreek.utils.config")
    cfg_stub.load_config = lambda path=None: {}
    sys.modules["datacreek.utils.config"] = cfg_stub

    pc = types.ModuleType("prometheus_client")

    class _DummyCounter:
        def __init__(self, *a, **k):
            self._value = SimpleNamespace(get=lambda: 0)

        def inc(self):
            pass

    class _DummyGauge:
        def __init__(self, *a, **k):
            pass

        def set(self, val):
            pass

    pc.CollectorRegistry = object
    pc.Counter = _DummyCounter
    pc.Gauge = _DummyGauge
    sys.modules["prometheus_client"] = pc

    import datacreek.utils.cache as cache

    cache.ttl_manager = None  # type: ignore
    cache.TTLManager.start = lambda self: None  # type: ignore
    importlib.reload(cache)


class DummyCounter:
    def __init__(self):
        self.value = 0
        self._value = SimpleNamespace(get=self.get)

    def inc(self):
        self.value += 1

    def get(self):
        return self.value


class DummyGauge:
    def __init__(self):
        self.value = None

    def set(self, val):
        self.value = val


class FakeRedis:
    class RedisError(Exception):
        pass

    def __init__(self):
        self.store = {}

    def exists(self, key):
        return key in self.store

    def get(self, key):
        return self.store[key]

    def setex(self, key, ttl, value):
        self.store[key] = value


def test_run_once_adjusts_ttl(monkeypatch):
    import datacreek.utils.cache as cache

    monkeypatch.setattr(cache.TTLManager, "start", lambda self: None)
    cache = importlib.reload(cache)
    hits = DummyCounter()
    miss = DummyCounter()
    gauge = DummyGauge()
    monkeypatch.setattr(cache, "hits", hits)
    monkeypatch.setattr(cache, "miss", miss)
    monkeypatch.setattr(cache, "hit_ratio_g", gauge)

    manager = cache.TTLManager()
    hits.inc()
    hits.inc()
    miss.inc()
    manager.run_once()
    assert gauge.value is not None
    assert manager.current_ttl < 3600


def test_l1_cache_decorator(monkeypatch):
    import datacreek.utils.cache as cache

    monkeypatch.setattr(cache.TTLManager, "start", lambda self: None)
    cache = importlib.reload(cache)

    fake = FakeRedis()
    monkeypatch.setattr(cache, "redis", fake)
    monkeypatch.setattr(cache, "hits", DummyCounter())
    monkeypatch.setattr(cache, "miss", DummyCounter())
    monkeypatch.setattr(cache, "hit_ratio_g", DummyGauge())
    monkeypatch.setattr(cache, "ttl_manager", cache.TTLManager())

    @cache.l1_cache(lambda x: str(x))
    def square(x):
        return x * x

    assert square(3) == 9
    assert fake.store["3"] == 9
    assert square(3) == 9  # hit path


def test_cache_l1_injects_client(monkeypatch):
    import datacreek.utils.cache as cache

    monkeypatch.setattr(cache.TTLManager, "start", lambda self: None)
    cache = importlib.reload(cache)

    fake = FakeRedis()

    class StubRedis:
        def Redis(self):
            return fake

    monkeypatch.setattr(cache, "redis", StubRedis())

    @cache.cache_l1
    def load(key, *, redis_client=None):
        assert redis_client is fake
        return "ok"

    assert load("k") == "ok"
