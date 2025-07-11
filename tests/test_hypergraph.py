import numpy as np

from datacreek.analysis.hypergraph import hyper_sagnn_embeddings, hyper_sagnn_head_drop_embeddings
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_hyper_sagnn_embeddings_shape():
    edges = [[0, 1], [1, 2, 3]]
    feats = np.arange(12).reshape(4, 3).astype(float)
    emb = hyper_sagnn_embeddings(edges, feats, embed_dim=4, seed=0)
    assert emb.shape == (2, 4)


def test_dataset_hyper_sagnn_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "a", "x")
    ds.add_chunk("d", "b", "y")
    ds.add_chunk("d", "c", "z")
    ds.add_hyperedge("h1", ["a", "b", "c"])

    for idx, n in enumerate(["a", "b", "c"]):
        ds.graph.graph.nodes[n]["embedding"] = [float(idx), 0.0]

    result = ds.compute_hyper_sagnn_embeddings(embed_dim=2, seed=0)
    assert "h1" in result
    assert len(result["h1"]) == 2
    assert any(e.operation == "compute_hyper_sagnn_embeddings" for e in ds.events)

    result_hd = ds.compute_hyper_sagnn_head_drop_embeddings(num_heads=2, threshold=0.0, seed=0)
    assert "h1" in result_hd
    assert len(result_hd["h1"]) == 1
    assert any(e.operation == "compute_hyper_sagnn_head_drop_embeddings" for e in ds.events)


def test_hyper_sagnn_head_drop():
    edges = [[0, 1, 2], [2, 3]]
    feats = np.random.RandomState(0).randn(4, 4)
    emb = hyper_sagnn_head_drop_embeddings(edges, feats, num_heads=2, threshold=0.0, seed=0)
    assert emb.shape[0] == 2
    assert emb.shape[1] == 2
