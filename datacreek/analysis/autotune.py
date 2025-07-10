from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx
import numpy as np

from .fractal import bottleneck_distance, fractal_level_coverage
from .information import graph_information_bottleneck, mdl_description_length, structural_entropy


@dataclass
class AutoTuneState:
    """State of the autotuning loop.

    Attributes
    ----------
    tau:
        Triangle count threshold for :func:`structural_entropy`.
    beta:
        Weight of the information bottleneck regularizer.
    eps:
        Allowed additive error for :func:`bottleneck_distance`.
    delta:
        MDL tolerance used when selecting motifs.
    p:
        Return bias of the Node2Vec random walk.
    q:
        In-out bias of the Node2Vec random walk.
    dim:
        Dimension of the Node2Vec embeddings.
    prev_graph:
        Previous graph snapshot for distance computations.
    """

    tau: int = 1
    beta: float = 0.1
    eps: float = 0.05
    delta: float = 0.05
    p: float = 1.0
    q: float = 1.0
    dim: int = 64
    prev_graph: Optional[nx.Graph] = None
    coverage_min: float = 0.0


def autotune_step(
    graph: nx.Graph,
    embeddings: Dict[object, Iterable[float]],
    labels: Dict[object, int],
    motifs: Iterable[nx.Graph],
    state: AutoTuneState,
    *,
    weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    lr: float = 0.1,
) -> Dict[str, Any]:
    """Perform one step of the autotuning procedure.

    The function measures four objectives on ``graph`` using ``state``:
    structural entropy ``H`` (after filtering triangle counts),
    bottleneck distance ``D`` to the previous graph, information bottleneck
    loss ``I`` on ``embeddings``/``labels`` and MDL value ``M`` over ``motifs``.
    A weighted cost ``J`` is computed and ``state`` is updated via simple
    gradient heuristics on ``tau`` and ``eps``.

    Parameters
    ----------
    graph:
        Current knowledge graph snapshot.
    embeddings:
        Mapping of node to embedding vector used by the information bottleneck.
    labels:
        Mapping of node to integer class label.
    motifs:
        Candidate subgraphs used for the MDL score.
    state:
        Mutable autotuning state.
    weights:
        Weights ``(w1, w2, w3, w4, w5)`` of the multi-objective cost. ``w5``
        controls the penalty on the variance of the Node2Vec norms.
    lr:
        Learning rate for the gradient heuristics.

    Returns
    -------
    dict
        Dictionary with measured metrics and updated parameters.
    """

    H = structural_entropy(graph, state.tau)
    coverage = fractal_level_coverage(graph)
    if state.prev_graph is not None:
        D = bottleneck_distance(state.prev_graph, graph, approx_epsilon=state.eps)
    else:
        D = 0.0
    I = graph_information_bottleneck(embeddings, labels, beta=state.beta)
    M = mdl_description_length(graph, motifs, delta=state.delta)

    norms = [np.linalg.norm(v) for v in embeddings.values()] if embeddings else [0.0]
    var_phi = float(np.var(norms))

    J = (
        weights[0] * (-H)
        + weights[1] * D
        + weights[2] * I
        + weights[3] * M
        + weights[4] * (-var_phi)
    )

    # finite difference gradients for tau, eps, beta and delta
    H_next = structural_entropy(graph, state.tau + 1)
    grad_tau = H_next - H
    state.tau = max(1, int(state.tau - lr * grad_tau))

    if state.prev_graph is not None:
        D_next = bottleneck_distance(state.prev_graph, graph, approx_epsilon=state.eps + 0.01)
        grad_eps = (D_next - D) / 0.01
        state.eps = max(0.0, state.eps + lr * grad_eps)

    I_next = graph_information_bottleneck(embeddings, labels, beta=state.beta + 0.01)
    grad_beta = (I_next - I) / 0.01
    state.beta = max(0.0, state.beta - lr * grad_beta)

    M_next = mdl_description_length(graph, motifs, delta=state.delta + 0.01)
    grad_delta = (M_next - M) / 0.01
    state.delta = max(0.0, state.delta - lr * grad_delta)

    state.prev_graph = graph.copy()

    if coverage < state.coverage_min:
        # slightly loosen the triangle threshold when coverage is too low
        state.tau += 1

    return {
        "entropy": H,
        "bottleneck": D,
        "ib_loss": I,
        "mdl": M,
        "coverage": coverage,
        "var_phi": var_phi,
        "cost": J,
        "tau": state.tau,
        "eps": state.eps,
        "beta": state.beta,
        "delta": state.delta,
    }
