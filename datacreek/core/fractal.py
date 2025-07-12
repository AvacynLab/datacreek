import logging
import random
from typing import Iterable

import numpy as np

from ..analysis.fractal import colour_box_dimension
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def bootstrap_sigma_db(graph: KnowledgeGraph, radii: Iterable[int]) -> float:
    """Estimate σ_{d_B} via bootstrap sampling.

    Parameters
    ----------
    graph: KnowledgeGraph
        Graph object to sample from.
    radii: Iterable[int]
        Box radii used when estimating ``d_B``.

    Returns
    -------
    float
        Bootstrap standard deviation of fractal dimension estimates.
    """
    dims = []
    nodes = list(graph.graph.nodes())
    for _ in range(30):
        sampled = graph.graph.subgraph(
            random.sample(nodes, max(1, int(0.8 * len(nodes))))
        )
        dim, _ = colour_box_dimension(sampled, radii)
        dims.append(dim)
    if not dims:
        return 0.0
    mean = float(np.mean(dims))
    sigma = float(np.sqrt(sum((d - mean) ** 2 for d in dims) / max(1, len(dims) - 1)))
    graph.graph.graph["fractal_sigma"] = sigma
    logger.info("fractal_sigma=%.4f", sigma)
    return sigma
