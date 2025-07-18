from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from skopt import Optimizer

from .index import recall10


def _var_norm(graph: "KnowledgeGraph") -> float:
    """Return variance of embedding norms stored on ``graph``."""
    embs = [
        np.asarray(data["embedding"], dtype=float)
        for _, data in graph.graph.nodes(data=True)
        if "embedding" in data
    ]
    if not embs:
        return float("inf")
    norms = np.linalg.norm(np.vstack(embs), axis=1)
    return float(np.var(norms))


def autotune_node2vec(
    kg: "KnowledgeGraph",
    queries: Sequence[object],
    ground_truth: Dict[object, Sequence[object]],
    *,
    k: int = 10,
    var_threshold: float = 1e-4,
    max_evals: int = 40,
) -> tuple[float, float]:
    """Tune Node2Vec ``p`` and ``q`` using Bayesian optimisation."""

    opt = Optimizer([(0.1, 4.0), (0.1, 4.0)], base_estimator="GP", acq_func="EI")
    best_recall = -1.0
    best_pq = (1.0, 1.0)

    for _ in range(max_evals):
        cand = opt.ask()
        p, q = float(cand[0]), float(cand[1])
        kg.compute_node2vec_embeddings(p=p, q=q)
        rec = recall10(kg.graph, queries, ground_truth, gamma=0.5, eta=0.25)
        var_n = _var_norm(kg)
        opt.tell(cand, -rec)
        if rec > best_recall:
            best_recall = rec
            best_pq = (p, q)
        if var_n < var_threshold:
            break

    kg.compute_node2vec_embeddings(p=best_pq[0], q=best_pq[1])
    recall10(kg.graph, queries, ground_truth, gamma=0.5, eta=0.25)
    return best_pq
