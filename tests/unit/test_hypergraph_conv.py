import numpy as np
import pytest

from datacreek.analysis.hypergraph_conv import (
    hypergraph_laplacian,
    estimate_lambda_max,
)


def build_simple_laplacian():
    """Construct a small hypergraph Laplacian for testing."""
    # Incidence matrix for 3 nodes and 2 hyperedges
    # Edge e0 connects v0 and v1; e1 connects all three nodes
    B = np.array(
        [
            [1, 1],
            [1, 1],
            [0, 1],
        ],
        dtype=float,
    )
    return hypergraph_laplacian(B)


def test_power_iteration_estimate_close_to_true():
    """Power iteration should approximate the dominant eigenvalue within 5%."""
    Delta = build_simple_laplacian()
    true = np.linalg.eigvalsh(Delta).max()
    est = estimate_lambda_max(Delta, it=10)
    assert est == pytest.approx(true, rel=0.05)


def test_zero_matrix_returns_zero():
    """Zero Laplacian should yield an eigenvalue estimate of zero."""
    Delta = np.zeros((3, 3))
    assert estimate_lambda_max(Delta) == pytest.approx(0.0)
