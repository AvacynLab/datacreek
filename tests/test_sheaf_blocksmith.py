import networkx as nx
import numpy as np
import pytest

sympy = pytest.importorskip("sympy", reason="sympy not installed")

from datacreek.analysis.sheaf import block_smith, validate_section


def test_block_smith_rank():
    L = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    rank = block_smith(L, block_size=2)
    assert isinstance(rank, int) and rank >= 0


def test_validate_section_score():
    g = nx.path_graph(3)
    score = validate_section(g, [0, 1])
    assert 0.0 <= score <= 1.0
