from datacreek.server.app import app, DATASETS
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType
from datacreek.core.knowledge_graph import KnowledgeGraph

import datacreek.server.app as app_module

def test_dataset_graph_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "text")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        res = client.get("/datasets/demo/graph")
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
    DATASETS.clear()


def test_dataset_search_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello world")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        res = client.get("/datasets/demo/search", query_string={"q": "hello"})
        assert res.status_code == 200
        data = res.get_json()
        assert data == ["c1"]
    DATASETS.clear()


def test_dataset_ingest_route(tmp_path):
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    DATASETS["demo"] = ds

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    with app.test_client() as client:
        res = client.post(
            "/datasets/demo/ingest",
            data={"input_path": str(f), "doc_id": "doc1"},
            follow_redirects=True,
        )
        assert res.status_code == 200
        assert ds.search_chunks("hello") == ["doc1_chunk_0"]
    DATASETS.clear()


def test_save_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    called = {}
    def fake_to_neo4j(self, driver, clear=True):
        called['called'] = True
    monkeypatch.setattr(KnowledgeGraph, "to_neo4j", fake_to_neo4j)

    class DummyDriver:
        def close(self):
            pass
    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        res = client.post("/datasets/demo/save_neo4j")
        assert res.status_code == 302
    assert called.get('called')
    DATASETS.clear()


def test_load_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    new_graph = KnowledgeGraph()
    new_graph.add_document("n", source="a")
    monkeypatch.setattr(KnowledgeGraph, "from_neo4j", staticmethod(lambda driver: new_graph))

    class DummyDriver:
        def close(self):
            pass
    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        res = client.post("/datasets/demo/load_neo4j")
        assert res.status_code == 302
    assert ds.graph is new_graph
    DATASETS.clear()
