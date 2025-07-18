import importlib.abc
import importlib.util
import time
from pathlib import Path

import networkx as nx
import pytest

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


class DummyCopy:
    def __init__(self, sql):
        self.sql = sql
        self.rows = []

    def write_row(self, row):
        self.rows.append(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyCursor:
    def __init__(self):
        self.calls = []
        self.copies = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def copy(self, sql):
        cp = DummyCopy(sql)
        self.copies.append(cp)
        return cp

    def fetchall(self):
        return []

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
    rows = pgvector_export.export_embeddings_pg(kg, conn, table="emb", lists=10)
    duration = time.perf_counter() - start

    assert rows == 4
    assert conn.committed
    assert duration < 0.03
    assert any("COPY emb" in cp.sql for cp in conn.cur.copies)
    assert any("ivfflat" in sql for sql, _ in conn.cur.calls)


def test_query_topk_sql():
    conn = DummyConn()
    vec = [0.1, 0.2]
    pgvector_export.query_topk_pg(conn, table="emb", vec=vec, k=3)
    assert any("SELECT node_id, space FROM emb" in sql for sql, _ in conn.cur.calls)


@pytest.mark.faiss_gpu
def test_recall_vs_faiss():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((200, 4)).astype("float32")
    xq = rng.standard_normal((10, 4)).astype("float32")
    index = faiss.IndexFlatIP(4)
    index.add(xb)
    _, gt = index.search(xq, 5)
    results = []
    for q in xq:
        sims = xb @ q
        ids = np.argsort(sims)[::-1][:5]
        results.append(ids)
    recall = sum(len(set(r).intersection(g)) for r, g in zip(results, gt)) / (
        len(xq) * 5
    )
    assert recall == 1.0
