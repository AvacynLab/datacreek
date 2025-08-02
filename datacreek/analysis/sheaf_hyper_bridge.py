from __future__ import annotations

"""Bridge between sheaf and hypergraph spectra."""

import networkx as nx
import numpy as np

from .monitoring import update_metric
from .sheaf import sheaf_incidence_matrix, sheaf_laplacian


def sheaf_hyper_incidence(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign"
) -> np.ndarray:
    """Return signed incidence matrix from a sheaf graph."""
    return sheaf_incidence_matrix(graph, edge_attr=edge_attr)


def sheaf_hyper_bridge_score(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign"
) -> float:
    r"""Return spectral coherence score between sheaf and hypergraph views.

    The score compares eigenvalues of the sheaf Laplacian and the hypergraph
    Laplacian constructed from the signed incidence matrix of ``graph``.
    It is defined as

    .. math::

       S = 1 - \frac{\sum_i |\lambda_i^{\text{sheaf}} - \lambda_i^{\text{hyper}}|}
                {\sum_i \lambda_i^{\text{sheaf}} + \lambda_i^{\text{hyper}}}.

    Parameters
    ----------
    graph:
        Input graph with sheaf sign information.
    edge_attr:
        Name of the edge attribute storing the sheaf sign.

    Returns
    -------
    float
        Coherence score between 0 and 1 (higher is better).
    """

    B = sheaf_incidence_matrix(graph, edge_attr=edge_attr)
    if B.size == 0:
        return 0.0
    L_s = sheaf_laplacian(graph, edge_attr=edge_attr)
    L_h = B @ B.T
    eig_s = np.linalg.eigvalsh(L_s)
    eig_h = np.linalg.eigvalsh(L_h)
    k = min(len(eig_s), len(eig_h))
    diff = float(np.sum(np.abs(eig_s[:k] - eig_h[:k])))
    total = float(np.sum(eig_s[:k]) + np.sum(eig_h[:k]) + 1e-12)
    score = 1.0 - diff / total
    try:
        update_metric("sheaf_hyper_score", score)
    except Exception:
        pass  # metric or monitoring not available
    return score
