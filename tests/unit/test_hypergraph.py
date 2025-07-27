import numpy as np
import pytest

from datacreek.analysis import hypergraph


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(0)
    return rng.normal(size=(5, 3))


@pytest.fixture
def sample_edges():
    return [[0, 1], [1, 2, 3], [3, 4]]


def test_hyper_sagnn_embeddings_deterministic(sample_edges, sample_features):
    emb1 = hypergraph.hyper_sagnn_embeddings(sample_edges, sample_features, embed_dim=2, seed=42)
    emb2 = hypergraph.hyper_sagnn_embeddings(sample_edges, sample_features, embed_dim=2, seed=42)
    assert emb1.shape == (len(sample_edges), 2)
    assert np.allclose(emb1, emb2)


def test_hyper_sagnn_head_drop_all_zero(sample_edges, sample_features):
    embs = hypergraph.hyper_sagnn_head_drop_embeddings(sample_edges, sample_features, num_heads=2, threshold=1.0, seed=0)
    assert embs.shape == (len(sample_edges), sample_features.shape[1] // 2)
    assert np.allclose(embs, 0)


def test_hyper_adamic_adar_scores():
    edges = [[1, 2, 3], [2, 3, 4]]
    scores = hypergraph.hyper_adamic_adar_scores(edges)
    w = 1.0 / np.log(3 - 1)
    assert pytest.approx(scores[(1, 2)], rel=1e-6) == w
    assert pytest.approx(scores[(1, 3)], rel=1e-6) == w
    assert pytest.approx(scores[(2, 3)], rel=1e-6) == 2 * w
    assert pytest.approx(scores[(2, 4)], rel=1e-6) == w
    assert pytest.approx(scores[(3, 4)], rel=1e-6) == w


def test_hyperedge_attention_scores_deterministic(sample_edges, sample_features):
    s1 = hypergraph.hyperedge_attention_scores(sample_edges, sample_features, seed=0)
    s2 = hypergraph.hyperedge_attention_scores(sample_edges, sample_features, seed=0)
    assert s1.shape == (len(sample_edges),)
    assert np.allclose(s1, s2)
    assert np.var(s1) > 0
