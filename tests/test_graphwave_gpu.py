import importlib

import networkx as nx
import numpy as np
import pytest

spec = importlib.util.find_spec("cupy")


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_chebyshev_heat_kernel_gpu_matches_cpu():
    import cupy as cp

    from datacreek.analysis.fractal import chebyshev_heat_kernel
    from datacreek.analysis.graphwave_cuda import chebyshev_heat_kernel_gpu

    g = nx.path_graph(4)
    L = nx.normalized_laplacian_matrix(g).tocsr()
    h_cpu = chebyshev_heat_kernel(L, 0.5, m=3)
    h_gpu = chebyshev_heat_kernel_gpu(L, 0.5, m=3)
    assert np.allclose(h_cpu, h_gpu, atol=1e-4)


@pytest.mark.skipif(spec is None, reason="cupy required")
def test_graphwave_runner_gpu(monkeypatch):
    from datacreek.core.knowledge_graph import KnowledgeGraph
    from datacreek.core.runners import GraphWaveRunner

    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.0)
    monkeypatch.setattr(kg, "compute_graphwave_embeddings", lambda *a, **k: None)
    runner = GraphWaveRunner(kg)
    runner.run(scales=[0.5], gpu=True)
