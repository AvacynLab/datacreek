from typing import Dict

import numpy as np


def graph_information_bottleneck(
    features: Dict[object, np.ndarray],
    labels: Dict[object, int],
    *,
    beta: float = 1.0,
) -> float:
    """Return a simple Graph Information Bottleneck loss.

    Parameters
    ----------
    features:
        Mapping from node to embedding vector.
    labels:
        Mapping from node to integer label.
    beta:
        Weight of the information regularizer.

    Returns
    -------
    float
        Value of the information bottleneck objective.
    """

    nodes = [n for n in labels if n in features]
    if not nodes:
        raise ValueError("no features for provided labels")

    X = np.stack([features[n] for n in nodes])
    y = np.array([labels[n] for n in nodes])

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, n_jobs=1)
    model.fit(X, y)
    probs = model.predict_proba(X)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y]))

    cov = np.cov(X, rowvar=False)
    reg = 0.5 * np.log(np.linalg.det(np.eye(cov.shape[0]) + cov))

    return float(ce + beta * reg)


def prototype_subgraph(
    graph: "nx.Graph",
    features: Dict[object, np.ndarray],
    labels: Dict[object, int],
    class_id: int,
    *,
    radius: int = 1,
) -> "nx.Graph":
    """Return a prototype subgraph for ``class_id`` using a simple IB model.

    A logistic regression classifier is trained on ``features`` and ``labels``.
    The node with the highest predicted probability for ``class_id`` is chosen
    as the prototype center. The subgraph induced by nodes within ``radius``
    hops of this center is returned.
    """

    import networkx as nx

    nodes = [n for n in labels if n in features]
    if not nodes:
        raise ValueError("no features for provided labels")

    X = np.stack([features[n] for n in nodes])
    y = np.array([labels[n] for n in nodes])

    model = LogisticRegression(max_iter=1000, n_jobs=1)
    model.fit(X, y)
    probs = model.predict_proba(X)

    if class_id >= probs.shape[1]:
        raise ValueError("class_id out of range")

    center_idx = int(np.argmax(probs[:, class_id]))
    center = nodes[center_idx]

    neighborhood = nx.single_source_shortest_path_length(graph, center, cutoff=radius)
    sub = graph.subgraph(neighborhood.keys()).copy()
    return sub

import math
import networkx as nx
from typing import Iterable, List


def mdl_description_length(graph: nx.Graph, motifs: Iterable[nx.Graph]) -> float:
    """Return a simplified MDL description length for ``motifs`` in ``graph``."""
    edges = {tuple(sorted(e)) for e in graph.edges()}
    base_bits = math.log(len(edges) + 1.0)
    motif_bits = math.log(graph.number_of_nodes() + 1.0)
    covered: set[tuple[int, int]] = set()
    for m in motifs:
        covered.update(tuple(sorted(e)) for e in m.edges())
    residual = len(edges - covered)
    return len(list(motifs)) * motif_bits + residual * base_bits


def select_mdl_motifs(graph: nx.Graph, motifs: Iterable[nx.Graph]) -> List[nx.Graph]:
    """Greedy selection of motifs lowering MDL of ``graph``."""
    motif_list = list(motifs)
    edges = {tuple(sorted(e)) for e in graph.edges()}
    base_bits = math.log(len(edges) + 1.0)
    motif_bits = math.log(graph.number_of_nodes() + 1.0)
    selected: List[nx.Graph] = []
    uncovered = set(edges)

    while True:
        best_idx = None
        best_gain = 0.0
        best_edges: set[tuple[int, int]] | None = None
        for i, m in enumerate(motif_list):
            m_edges = {tuple(sorted(e)) for e in m.edges()}
            cover = len(m_edges & uncovered)
            gain = cover * base_bits - motif_bits
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_edges = m_edges
        if best_idx is None or best_gain <= 0:
            break
        selected.append(motif_list.pop(best_idx))
        uncovered -= best_edges  # type: ignore[arg-type]
    return selected




def graph_entropy(graph: nx.Graph, *, base: float = 2.0) -> float:
    """Return Shannon entropy of the node degree distribution."""
    degrees = [d for _, d in graph.degree()]
    if not degrees:
        return 0.0
    vals, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    logp = np.log(probs) / np.log(base)
    return float(-np.sum(probs * logp))


def subgraph_entropy(graph: nx.Graph, nodes: Iterable, *, base: float = 2.0) -> float:
    """Return entropy of the degree distribution inside ``nodes``.

    Parameters
    ----------
    graph:
        Whole graph containing the subgraph.
    nodes:
        Iterable of nodes defining the subgraph.
    base:
        Logarithm base for entropy computation.

    Returns
    -------
    float
        Shannon entropy of the degree distribution in the induced subgraph.
    """
    sub = graph.subgraph(nodes)
    return graph_entropy(sub, base=base)
