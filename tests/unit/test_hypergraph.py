import numpy as np
import pytest

from datacreek.analysis.hypergraph import (
    hyper_sagnn_embeddings,
    hyper_sagnn_head_drop_embeddings,
    hyper_adamic_adar_scores,
    hyperedge_attention_scores,
)


def test_hyper_sagnn_embeddings_deterministic_and_shape():
    node_features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = [[0, 1], [1, 2]]
    emb1 = hyper_sagnn_embeddings(edges, node_features, seed=0)
    emb2 = hyper_sagnn_embeddings(edges, node_features, seed=0)
    assert emb1.shape == (2, 2)
    assert np.allclose(emb1, emb2)

    emb3 = hyper_sagnn_embeddings(edges, node_features, embed_dim=3, seed=0)
    assert emb3.shape == (2, 3)


def test_hyper_sagnn_head_drop_embeddings_threshold_behavior():
    node_features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = [[0, 1], [1, 2]]
    # threshold low enough to keep heads
    out = hyper_sagnn_head_drop_embeddings(edges, node_features, num_heads=2, threshold=0.0, seed=0)
    assert out.shape == (2, 1)
    assert not np.allclose(out, 0)

    # high threshold drops all heads -> zeros
    out2 = hyper_sagnn_head_drop_embeddings(edges, node_features, num_heads=2, threshold=1.0, seed=0)
    assert np.allclose(out2, 0)


def test_hyper_adamic_adar_scores_expected_values():
    edges = [[1, 2, 3], [2, 3], [3, 4]]
    scores = hyper_adamic_adar_scores(edges)
    approx = pytest.approx
    val = 1 / np.log(2)
    assert scores[(1, 2)] == approx(val)
    assert scores[(1, 3)] == approx(val)
    assert scores[(2, 3)] == approx(val)
    assert scores[(3, 4)] == 0.0


def test_hyperedge_attention_scores_deterministic():
    node_features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = [[0, 1], [1, 2]]
    scores = hyperedge_attention_scores(edges, node_features, seed=0)
    assert scores.shape == (2,)
    assert np.allclose(scores, [0.5, 0.5])
