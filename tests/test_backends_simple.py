import os
import sys
from types import SimpleNamespace

import datacreek.backends as backends
import datacreek.storage as storage


def test_get_redis_client_none(monkeypatch):
    backends.get_redis_client.cache_clear()
    monkeypatch.setattr(backends, "redis", None)
    assert backends.get_redis_client() is None


def test_get_redis_client_env(monkeypatch):
    backends.get_redis_client.cache_clear()
    class FakeRedis:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.setattr(backends, "redis", SimpleNamespace(Redis=FakeRedis))
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setattr(backends, "get_redis_config", lambda cfg: {"host": "h", "port": 5})
    os.environ["REDIS_HOST"] = "host"
    os.environ["REDIS_PORT"] = "10"
    client = backends.get_redis_client()
    assert isinstance(client, FakeRedis)
    assert client.kw["host"] == "host" and client.kw["port"] == 10
    os.environ.pop("REDIS_HOST")
    os.environ.pop("REDIS_PORT")


def test_get_neo4j_driver_missing(monkeypatch):
    backends.get_neo4j_driver.cache_clear()
    monkeypatch.setattr(backends, "GraphDatabase", None)
    assert backends.get_neo4j_driver() is None


def test_get_neo4j_driver_success(monkeypatch):
    backends.get_neo4j_driver.cache_clear()
    class FakeSession:
        def __init__(self):
            self.executed = []
        def run(self, stmt):
            self.executed.append(stmt)
    class FakeDriver:
        def __init__(self):
            self.session_obj = FakeSession()
        def session(self):
            return self.session_obj
    class FakeGraphDB:
        def driver(self, uri, auth=None):
            self.uri = uri
            self.auth = auth
            return FakeDriver()
    fake_graphdb = FakeGraphDB()
    monkeypatch.setattr(backends, "GraphDatabase", fake_graphdb)
    monkeypatch.setattr(backends, "ensure_neo4j_indexes", lambda d: d.session().run("index"))
    monkeypatch.setattr(backends, "run_cypher_file", lambda d, f: d.session().run(f))
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setattr(backends, "get_neo4j_config", lambda cfg: {"uri": "u", "user": "u", "password": "p", "run_migrations": True})
    os.environ.update({"NEO4J_URI": "bolt://x", "NEO4J_USER": "n", "NEO4J_PASSWORD": "s", "NEO4J_INIT_INDEXES": "1"})
    driver = backends.get_neo4j_driver()
    assert isinstance(driver, FakeDriver)
    assert driver.session_obj.executed == ["index", "2025-07-haa_index.cypher", "2025-07-haa_unique_constraint.cypher"]
    for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_INIT_INDEXES"]:
        os.environ.pop(k, None)


def test_get_redis_graph(monkeypatch):
    backends.get_redis_graph.cache_clear()
    class FakeRG:
        def __init__(self, name, client):
            self.name = name
            self.client = client
    class FakeRedis:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.setattr(backends, "RedisGraph", FakeRG)
    monkeypatch.setattr(backends, "redis", SimpleNamespace(Redis=FakeRedis))
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setattr(backends, "get_redis_config", lambda cfg: {"host": "h", "port": 1})
    os.environ["USE_REDIS_GRAPH"] = "1"
    g = backends.get_redis_graph("A")
    assert isinstance(g, FakeRG)
    assert isinstance(g.client, FakeRedis)
    os.environ.pop("USE_REDIS_GRAPH")


def test_get_s3_storage(monkeypatch):
    class FakeClient:
        def __init__(self):
            pass
    class FakeBoto:
        def client(self, name, **kw):
            return FakeClient()
    fake_boto = FakeBoto()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto)
    monkeypatch.setattr(backends, "boto3", fake_boto)
    os.environ.update({"S3_BUCKET": "b", "AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"})
    store = backends.get_s3_storage()
    assert store.bucket == "b"
    os.environ.pop("S3_BUCKET")
    os.environ.pop("AWS_ACCESS_KEY_ID")
    os.environ.pop("AWS_SECRET_ACCESS_KEY")


def test_ensure_neo4j_indexes(monkeypatch, tmp_path):
    executed = []
    class FakeSession:
        def run(self, stmt):
            executed.append(stmt)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    driver = SimpleNamespace(session=lambda: FakeSession())
    backends.ensure_neo4j_indexes(driver)
    assert executed  # all statements attempted


def test_run_cypher_file(monkeypatch, tmp_path):
    path = tmp_path / "file.cypher"
    path.write_text("CREATE X; CREATE Y;")
    executed = []
    class FakeSession:
        def run(self, stmt):
            executed.append(stmt)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    driver = SimpleNamespace(session=lambda: FakeSession())
    backends.run_cypher_file(driver, str(path))
    assert executed == ["CREATE X", "CREATE Y"]
    executed.clear()
    backends.run_cypher_file(driver, str(tmp_path / "missing.cypher"))
    assert executed == []

