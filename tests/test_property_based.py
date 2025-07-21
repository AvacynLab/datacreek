"""Property-based tests for mathematical invariants."""

import networkx as nx
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from datacreek.analysis.fractal import graphwave_embedding_chebyshev
from datacreek.analysis.poincare_recentering import _mobius_add


@given(st.integers(min_value=3, max_value=6))
@settings(max_examples=5)
def test_graphwave_symmetry(n: int) -> None:
    """Embeddings for path graphs should be symmetric."""
    g = nx.path_graph(n)
    emb = graphwave_embedding_chebyshev(g, [0.5], num_points=4, order=3)
    for i in range(n):
        j = n - 1 - i
        assert np.allclose(emb[i], emb[j])


vector = st.lists(st.floats(-0.4, 0.4), min_size=2, max_size=2).map(
    lambda l: np.array(l, dtype=float)
)


@given(x=vector, y=vector, z=vector)
@settings(max_examples=25)
def test_mobius_add_associative(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Möbius addition is approximately associative."""
    res1 = _mobius_add(_mobius_add(x, y, clamp=False), z, clamp=False)
    res2 = _mobius_add(x, _mobius_add(y, z, clamp=False), clamp=False)
    assert np.allclose(res1, res2, atol=5e-2)
