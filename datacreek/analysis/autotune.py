from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Sequence

import networkx as nx
import numpy as np

from .fractal import bottleneck_distance, fractal_level_coverage
from .information import (
    graph_information_bottleneck,
    mdl_description_length,
    structural_entropy,
)
from .multiview import hybrid_score


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
    alpha: float = 0.5
    gamma: float = 0.5
    eta: float = 0.25
    prev_graph: Optional[nx.Graph] = None
    coverage_min: float = 0.0


def recall_at_k(
    graph: nx.Graph,
    queries: Sequence[object],
    ground_truth: Dict[object, Sequence[object]],
    *,
    k: int = 10,
    gamma: float = 0.5,
    eta: float = 0.25,
) -> float:
    """Compute mean recall@k using the hybrid similarity score.

    Parameters
    ----------
    graph:
        Knowledge graph containing embeddings on nodes.
    queries:
        Nodes for which to compute retrieval performance.
    ground_truth:
        Mapping of query node to relevant nodes.
    k:
        Cutoff rank.
    gamma, eta:
        Weights used by :func:`hybrid_score`.
    """

    total = 0.0
    n = 0
    for q in queries:
        rel = set(ground_truth.get(q, []))
        if not rel:
            continue

        node_data = graph.nodes[q]
        n2v_q = node_data.get("embedding")
        gw_q = node_data.get("graphwave_embedding")
        hyp_q = node_data.get("poincare_embedding")
        if n2v_q is None or gw_q is None or hyp_q is None:
            continue

        scores = []
        for u, data in graph.nodes(data=True):
            if u == q:
                continue
            n2v_u = data.get("embedding")
            gw_u = data.get("graphwave_embedding")
            hyp_u = data.get("poincare_embedding")
            if n2v_u is None or gw_u is None or hyp_u is None:
                continue
            s = hybrid_score(n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q, gamma=gamma, eta=eta)
            scores.append((u, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [u for u, _ in scores[:k]]
        hits = len(rel.intersection(retrieved))
        total += hits / len(rel)
        n += 1

    return total / n if n else 0.0


def autotune_step(
    graph: nx.Graph,
    embeddings: Dict[object, Iterable[float]],
    labels: Dict[object, int],
    motifs: Iterable[nx.Graph],
    state: AutoTuneState,
    *,
    weights: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    recall_data: Optional[Tuple[Sequence[object], Dict[object, Sequence[object]]]] = None,
    k: int = 10,
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
        Weights ``(w1, w2, w3, w4, w5, w6)`` of the multi-objective cost. ``w5``
        controls the penalty on the variance of the Node2Vec norms and ``w6``
        weights the negative recall term.
    recall_data:
        Optional tuple ``(queries, ground_truth)`` for computing recall@k with
        the hybrid similarity score.
    k:
        Rank cutoff used in the recall metric.
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

    recall = 0.0
    if recall_data is not None:
        queries, gt = recall_data
        recall = recall_at_k(
            graph,
            queries,
            gt,
            k=k,
            gamma=state.gamma,
            eta=state.eta,
        )

    J = (
        weights[0] * (-H)
        + weights[1] * D
        + weights[2] * I
        + weights[3] * M
        + weights[4] * (-var_phi)
        + weights[5] * (-recall)
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
        "recall": recall,
        "cost": J,
        "tau": state.tau,
        "eps": state.eps,
        "beta": state.beta,
        "delta": state.delta,
    }
