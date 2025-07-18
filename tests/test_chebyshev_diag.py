import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.chebyshev_diag import chebyshev_diag_hutchpp


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("scipy") is None, reason="scipy required"
)
def test_chebyshev_diag_accuracy():
    import scipy.linalg as la

    g = nx.path_graph(6)
    L = nx.normalized_laplacian_matrix(g).toarray()
    t = 0.5
    exact = np.diag(la.expm(-t * L))
    approx = chebyshev_diag_hutchpp(L, t, order=5, samples=64, rng=0)
    mse = np.mean((approx - exact) ** 2)
    assert mse < 1e-3
