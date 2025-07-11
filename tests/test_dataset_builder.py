import pytest

try:
    from datacreek.core.dataset import DatasetBuilder
except Exception:  # pragma: no cover - optional dependency missing
    DatasetBuilder = None  # type: ignore


def test_dataset_rollback_wrapper(tmp_path):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()
    path = ds.rollback_gremlin_diff(output="tmp.diff")
    assert path.endswith("tmp.diff")


def test_dataset_sheaf_sla_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()
    mttr = ds.sheaf_checker_sla([0, 1800])
    assert mttr == 0.5


def test_dataset_prune_fractalnet_weights_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    db = DatasetBuilder()
    w = [0.1 * i for i in range(10)]
    out = db.prune_fractalnet_weights(w, ratio=0.3)
    assert len([x for x in out if x != 0]) == 3


def test_dataset_colour_box_dimension_wrapper():
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.add_node("a")
    ds.graph.add_node("b")
    ds.graph.add_edge("a", "b")
    dim, counts = ds.colour_box_dimension([1])
    assert isinstance(dim, float) and len(counts) == 1


def test_dataset_search_hybrid_wrapper(monkeypatch):
    if DatasetBuilder is None:
        pytest.skip("DatasetBuilder unavailable")
    ds = DatasetBuilder()

    def fake(self, query, k=5, node_type="chunk"):
        assert query == "hello" and k == 2 and node_type == "chunk"
        return ["a", "b"]

    monkeypatch.setattr(ds.graph.__class__, "search_hybrid", fake)
    res = ds.search_hybrid("hello", k=2)
    assert res == ["a", "b"]
