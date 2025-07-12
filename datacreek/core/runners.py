import logging
from dataclasses import dataclass
from typing import Iterable

from ..utils.config import load_config
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Node2VecRunner:
    """Run Node2Vec embeddings using configuration parameters."""

    graph: KnowledgeGraph

    def run(
        self,
        *,
        dimensions: int | None = None,
        walk_length: int = 10,
        num_walks: int = 50,
        workers: int = 1,
        seed: int = 0,
    ) -> None:
        cfg = load_config()
        emb_cfg = cfg.get("embeddings", {}).get("node2vec", {})
        dim_val = dimensions or int(emb_cfg.get("dimension", 128))
        p_val = float(emb_cfg.get("p", 1.0))
        q_val = float(emb_cfg.get("q", 1.0))
        self.graph.compute_node2vec_embeddings(
            dimensions=dim_val,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
            p=p_val,
            q=q_val,
        )


@dataclass
class GraphWaveRunner:
    """Run GraphWave embeddings with Chebyshev approximation."""

    graph: KnowledgeGraph

    def run(
        self,
        scales: Iterable[float],
        *,
        num_points: int = 10,
        order: int = 7,
    ) -> None:
        self.graph.compute_graphwave_embeddings(
            scales=scales,
            num_points=num_points,
            chebyshev_order=order,
        )
        gw_entropy = self.graph.graphwave_entropy()
        self.graph.graph["gw_entropy"] = gw_entropy
        logger.info("gw_entropy=%.4f", gw_entropy)


@dataclass
class PoincareRunner:
    """Generate Poincar\xe9 embeddings and detect crowding."""

    graph: KnowledgeGraph

    def run(
        self,
        *,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        from ..analysis.fractal import poincare_embedding

        emb = poincare_embedding(
            self.graph.graph.to_undirected(),
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        for node, vec in emb.items():
            self.graph.graph.nodes[node]["poincare_embedding"] = vec.tolist()
