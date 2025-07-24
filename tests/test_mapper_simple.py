import os
import pickle
import threading
import time
import types
import networkx as nx
import pytest

import datacreek.analysis.mapper as mapper

class DummyRedis:
    def __init__(self):
        self.store = {}
        self.expire_called = False
    def setex(self, key, ttl, data):
        self.store[key] = data
    def get(self, key):
        return self.store.get(key)
    def incr(self, name):
        self.store[name] = self.store.get(name, 0) + 1
    def config_set(self, *a, **kw):
        pass
    def expire(self, key, ttl):
        self.expire_called = True
        self.store[f"ttl_{key}"] = ttl

class DummyTxn:
    def __init__(self, env, write=False):
        self.env = env
        self.write = write
        self.pos = 0
    def __enter__(self):
        return self
    def __exit__(self, exc, val, tb):
        pass
    def put(self, key, value):
        self.env.store[key] = value
    def get(self, key):
        return self.env.store.get(key)
    def delete(self, key=None):
        if key is None and hasattr(self, "keys"):
            key = self.keys[self.pos]
        if key in self.env.store:
            del self.env.store[key]
            if hasattr(self, "keys") and key in self.keys:
                self.keys.remove(key)
            return True
        return False
    def cursor(self):
        self.keys = list(self.env.store.keys())
        self.pos = 0
        return self
    def first(self):
        self.pos = 0
        return bool(self.keys)
    def next(self):
        self.pos += 1
        return self.pos < len(self.keys)
    def key(self):
        return self.keys[self.pos]
    def value(self):
        return self.env.store[self.keys[self.pos]]

class DummyEnv:
    def __init__(self):
        self.store = {}
    def set_mapsize(self, size):
        self.mapsize = size
    def stat(self):
        return {"psize": 1}
    def info(self):
        return {"map_size": len(self.store)}
    def begin(self, write=False):
        return DummyTxn(self, write)
    def sync(self):
        pass
    def close(self):
        pass

class DummyLMDBModule(types.SimpleNamespace):
    def __init__(self):
        super().__init__(open=self._open)
        self.env = DummyEnv()
    def _open(self, *a, **kw):
        return self.env

class DummyThread:
    def __init__(self, target, args=(), daemon=True):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
    def start(self):
        self.started = True
    def join(self, timeout=None):
        self.target(*self.args)

def test_mapper_roundtrip():
    g = nx.path_graph(4)
    nerve, cover = mapper.mapper_nerve(g, radius=1)
    reconstructed = mapper.inverse_mapper(nerve, cover)
    assert set(g.edges()).issubset(reconstructed.edges())


def test_cache_cycle(monkeypatch, tmp_path):
    fake_redis = DummyRedis()
    fake_lmdb = DummyLMDBModule()
    monkeypatch.setattr(mapper, "redis", types.SimpleNamespace(Redis=lambda: fake_redis))
    monkeypatch.setattr(mapper, "lmdb", fake_lmdb)
    monkeypatch.setattr(mapper, "start_l2_eviction_thread", lambda *a, **kw: None)
    monkeypatch.setattr(mapper, "load_config", lambda: {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}})
    g = nx.path_graph(3)
    nerve1, cover1 = mapper.cache_mapper_nerve(g, 1, ssd_dir=tmp_path)
    nerve2, cover2 = mapper.cache_mapper_nerve(g, 1, ssd_dir=tmp_path)
    assert nerve1.edges() == nerve2.edges()
    assert cover1 == cover2
    assert fake_redis.get(next(iter(fake_redis.store)))


def test_adjust_ttl(monkeypatch):
    fake = DummyRedis()
    monkeypatch.setattr(mapper, "_redis_hits", 8)
    monkeypatch.setattr(mapper, "_redis_misses", 2)
    monkeypatch.setattr(mapper, "_last_ttl_eval", 0)
    monkeypatch.setattr(mapper, "_redis_ttl", 3600)
    monkeypatch.setattr(mapper, "os", types.SimpleNamespace(getloadavg=lambda: (0,0,0), cpu_count=lambda: 1))
    mapper._adjust_ttl(fake, "k")
    assert fake.expire_called


def test_l2_evict_once_and_delete(monkeypatch):
    env = DummyEnv()
    old = time.time() - 7200
    env.store[b"old"] = pickle.dumps((old, b"data"))
    env.store[b"new"] = pickle.dumps((time.time(), b"data"))
    log = []
    monkeypatch.setattr(mapper, "log_eviction", lambda k, t, c: log.append((k, c)))
    mapper._l2_evict_once(env, limit_mb=0, ttl_h=0)
    assert any(entry[0] == "old" for entry in log)
    assert mapper.delete_l2_entry(env, "new")


def test_start_stop_eviction_thread(monkeypatch):
    events = []
    monkeypatch.setattr(mapper, "_evict_worker", lambda *args: events.append(args))
    monkeypatch.setattr(mapper, "load_config", lambda: {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}})
    monkeypatch.setattr(threading, "Thread", DummyThread)
    mapper.start_l2_eviction_thread("db.mdb", interval=0.0)
    assert mapper._evict_thread.started
    mapper.stop_l2_eviction_thread()
    assert mapper._evict_thread is None
    assert events
