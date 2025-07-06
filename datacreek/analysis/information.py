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
