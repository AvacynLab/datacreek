import logging
import random
from typing import Iterable

import numpy as np

from ..analysis.fractal import colour_box_dimension
from .knowledge_graph import KnowledgeGraph

try:  # optional neo4j dependency
    from neo4j import Driver
except Exception:  # pragma: no cover - optional
    Driver = None

logger = logging.getLogger(__name__)


def bootstrap_sigma_db(
    graph: KnowledgeGraph,
    radii: Iterable[int],
    *,
    driver: Driver | None = None,
    dataset: str | None = None,
) -> float:
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
    dims: list[float] = []

    if driver is not None and Driver is not None:
        ds = dataset or "kg_bs_tmp"
        try:
            graph.to_neo4j(driver, dataset=ds, clear=True)
            node_q = "MATCH (n {dataset:$ds}) RETURN id(n) AS id"
            rel_q = "MATCH (n {dataset:$ds})-[r]->(m {dataset:$ds}) RETURN id(n) AS source, id(m) AS target"
            with driver.session() as session:
                session.run("CALL gds.graph.drop('kg_bs', false)")
                session.run(
                    "CALL gds.graph.project.cypher('kg_bs', $nq, $rq)",
                    nq=node_q,
                    rq=rel_q,
                    ds=ds,
                )
                for i in range(30):
                    seed = random.randint(0, 2**31 - 1)
                    res = session.run(
                        "CALL gds.beta.graph.sample('kg_bs', {sampleRate:0.8, randomSeed:$seed}) YIELD nodeIds",
                        seed=seed,
                    ).single()
                    node_ids = res.get("nodeIds", []) if res else []
                    if not node_ids:
                        continue
                    edges = session.run(
                        "MATCH (a {dataset:$ds})-[r]->(b {dataset:$ds}) WHERE id(a) IN $ids AND id(b) IN $ids RETURN a.id AS u, b.id AS v",
                        ds=ds,
                        ids=node_ids,
                    )
                    sub = graph.graph.__class__()
                    sub.add_nodes_from(node_ids)
                    for r in edges:
                        sub.add_edge(r["u"], r["v"])
                    dim, _ = colour_box_dimension(sub, radii)
                    dims.append(dim)
                session.run("CALL gds.graph.drop('kg_bs')")
                session.run("MATCH (n {dataset:$ds}) DETACH DELETE n", ds=ds)
        except Exception:
            logger.exception("gds sampling failed, falling back to NetworkX")

    if not dims:
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
