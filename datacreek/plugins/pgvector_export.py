"""Export graph embeddings to PostgreSQL using pgvector."""

from __future__ import annotations

from typing import Iterable

from psycopg import Connection


def _fmt_vector(vec: Iterable[float] | None) -> str | None:
    """Return vector literal for pgvector."""
    if vec is None:
        return None
    return "[" + ",".join(f"{float(v):g}" for v in vec) + "]"


def export_embeddings_pg(
    kg: "KnowledgeGraph",
    conn: Connection,
    *,
    table: str = "node_embeddings",
    n2v_attr: str = "embedding",
    poincare_attr: str = "poincare_embedding",
) -> int:
    """Export Node2Vec and Poincaré embeddings to PostgreSQL.

    Parameters
    ----------
    kg:
        Graph storing embeddings on its nodes.
    conn:
        Open psycopg connection to the database with pgvector extension.
    table:
        Destination table, created if missing.
    n2v_attr:
        Node attribute containing Node2Vec embeddings.
    poincare_attr:
        Node attribute containing Poincaré embeddings.

    Returns
    -------
    int
        Number of rows written.
    """

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "node TEXT PRIMARY KEY,"
            "node2vec VECTOR,"
            "poincare VECTOR"
            ")"
        )
        rows = 0
        for node, data in kg.graph.nodes(data=True):
            n2v = _fmt_vector(data.get(n2v_attr))
            pc = _fmt_vector(data.get(poincare_attr))
            if n2v is None and pc is None:
                continue
            cur.execute(
                f"INSERT INTO {table} (node, node2vec, poincare) VALUES (%s, %s, %s) "
                "ON CONFLICT(node) DO UPDATE SET node2vec=EXCLUDED.node2vec, "
                "poincare=EXCLUDED.poincare",
                (str(node), n2v, pc),
            )
            rows += 1
    conn.commit()
    return rows
