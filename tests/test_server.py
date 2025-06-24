from datacreek.server.app import app, DATASETS
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


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
