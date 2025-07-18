import importlib.abc
import importlib.util
import time
from pathlib import Path

import networkx as nx

spec = importlib.util.spec_from_file_location(
    "datacreek.plugins.pgvector_export",
    Path(__file__).resolve().parents[1]
    / "datacreek"
    / "plugins"
    / "pgvector_export.py",
)
pgvector_export = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(pgvector_export)


class DummyKG:
    def __init__(self):
        self.graph = nx.Graph()


class DummyCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyConn:
    def __init__(self):
        self.cur = DummyCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def test_export_embeddings_latency():
    kg = DummyKG()
    kg.graph.add_node("a", embedding=[0.1, 0.2], poincare_embedding=[0.3, 0.4])
    kg.graph.add_node("b", embedding=[0.5, 0.6], poincare_embedding=[0.7, 0.8])
    conn = DummyConn()

    start = time.perf_counter()
    rows = pgvector_export.export_embeddings_pg(kg, conn, table="emb")
    duration = time.perf_counter() - start

    assert rows == 2
    assert conn.committed
    assert duration < 0.03
    assert any("INSERT INTO emb" in sql for sql, _ in conn.cur.calls)
