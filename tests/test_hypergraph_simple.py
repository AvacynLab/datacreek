import numpy as np
import pytest

from datacreek.analysis.hypergraph import (
    hyper_adamic_adar_scores,
    hyper_sagnn_embeddings,
    hyper_sagnn_head_drop_embeddings,
    hyperedge_attention_scores,
)


def test_hyper_sagnn_embeddings_deterministic():
    edges = [[0, 1], [1, 2]]
    feats = np.arange(9).reshape(3, 3).astype(float)
    emb1 = hyper_sagnn_embeddings(edges, feats, embed_dim=2, seed=0)
    emb2 = hyper_sagnn_embeddings(edges, feats, embed_dim=2, seed=0)
    assert np.allclose(emb1, emb2)
    assert emb1.shape == (2, 2)


def test_hyper_sagnn_head_drop_embeddings_threshold():
    edges = [[0, 1, 2], [2, 1]]
    feats = np.arange(12).reshape(4, 3).astype(float)
    emb_drop = hyper_sagnn_head_drop_embeddings(
        edges, feats, num_heads=2, threshold=1.0, seed=1
    )
    assert np.allclose(emb_drop, 0)
    emb_keep = hyper_sagnn_head_drop_embeddings(
        edges, feats, num_heads=2, threshold=0.0, seed=1
    )
    assert emb_keep.shape[1] == max(1, feats.shape[1] // 2)


def test_hyper_adamic_adar_scores_triangle():
    edges = [["a", "b", "c"], ["b", "c"]]
    scores = hyper_adamic_adar_scores(edges)
    tri_weight = 1.0 / np.log(2)
    assert scores[("a", "b")] == pytest.approx(tri_weight)
    assert scores[("a", "c")] == pytest.approx(tri_weight)
    assert scores[("b", "c")] >= tri_weight


def test_hyperedge_attention_scores():
    edges = [[0, 1], [1, 2, 3]]
    feats = np.random.RandomState(0).randn(4, 2)
    scores = hyperedge_attention_scores(edges, feats, seed=0)
    assert scores.shape == (2,)
    assert (scores > 0).all()
