from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .fractal import bottleneck_distance, fractal_level_coverage
from .information import graph_information_bottleneck, mdl_description_length, structural_entropy
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


def kw_gradient(f, x: float, h: float = 1.0, n: int = 4) -> float:
    """Return Kiefer-Wolfowitz stochastic gradient estimate.

    Parameters
    ----------
    f:
        Function taking a float and returning the scalar objective.
    x:
        Current parameter value.
    h:
        Perturbation size.
    n:
        Number of random evaluations.
    """
    rng = np.random.default_rng()
    grad = 0.0
    for _ in range(n):
        s = rng.choice([-1.0, 1.0])
        grad += s * f(x + s * h)
    return grad / (n * h)


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

    # stochastic KW gradients for tau and eps
    grad_tau = kw_gradient(lambda t: structural_entropy(graph, int(max(1, t))), state.tau)
    state.tau = max(1, int(state.tau - lr * grad_tau))

    if state.prev_graph is not None:
        grad_eps = kw_gradient(
            lambda e: bottleneck_distance(state.prev_graph, graph, approx_epsilon=max(0.0, e)),
            state.eps,
            h=0.01,
        )
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


def svgp_ei_propose(
    params: Sequence[Sequence[float]],
    scores: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    m: int = 100,
    n_samples: int = 256,
) -> np.ndarray:
    """Return new parameter vector maximizing Expected Improvement.

    A sparse Gaussian Process regression model is fitted on ``params`` and
    ``scores``. ``m`` subsamples are used as inducing points. ``n_samples``
    random candidates are drawn within ``bounds`` and the one with highest
    expected improvement over the best observed score is returned.
    """

    from math import inf

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C

    X = np.asarray(list(params), dtype=float)
    y = np.asarray(list(scores), dtype=float)
    if len(X) == 0:
        raise ValueError("no observations provided")
    idx = np.linspace(0, len(X) - 1, min(len(X), m), dtype=int)
    X_sub = X[idx]
    y_sub = y[idx]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_sub, y_sub)

    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    rng = np.random.default_rng()
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dim))

    y_best = y.min()
    best_x = None
    best_ei = -inf
    for x in candidates:
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        sigma = float(sigma) + 1e-9
        improvement = y_best - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        if ei > best_ei:
            best_ei = float(ei)
            best_x = x
    return best_x if best_x is not None else candidates[0]
