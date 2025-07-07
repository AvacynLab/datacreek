from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


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
