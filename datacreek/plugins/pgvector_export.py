"""Export graph embeddings to PostgreSQL using pgvector.

This module exposes helpers to bulk COPY embeddings into a table and build a
vector index for fast similarity search.  The schema follows the audit
specification::

    CREATE TABLE embedding (
        node_id UUID,
        space TEXT,
        vec VECTOR(dim)
    );

``export_embeddings_pg`` inserts the embeddings using ``COPY`` for better
performance and optionally creates an ``ivfflat`` index.
"""

from __future__ import annotations

from typing import Iterable, Sequence

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
    table: str = "embedding",
    attrs: Iterable[str] = ("embedding", "poincare_embedding"),
    lists: int = 100,
) -> int:
    """Export embeddings using ``COPY`` and build an ``ivfflat`` index.

    Parameters
    ----------
    kg:
        Graph storing embeddings on its nodes.
    conn:
        Open psycopg connection to the database with pgvector extension.
    table:
        Destination table, created if missing.
    attrs:
        Node attributes storing different embedding spaces.
    lists:
        Number of lists for the ``ivfflat`` index.

    Returns
    -------
    int
        Number of rows written.
    """

    rows = []
    dim = None
    for node, data in kg.graph.nodes(data=True):
        for space in attrs:
            vec = data.get(space)
            if vec is None:
                continue
            if dim is None:
                dim = len(vec)
            rows.append((str(node), space, _fmt_vector(vec)))

    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "node_id UUID,"
            "space TEXT,"
            f"vec VECTOR({dim})"
            ")"
        )
        with cur.copy(f"COPY {table} (node_id, space, vec) FROM STDIN") as cp:
            for row in rows:
                cp.write_row(row)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_ivfflat ON {table} USING ivfflat (vec) WITH (lists=%s)",
            (lists,),
        )
    conn.commit()
    return len(rows)


def query_topk_pg(conn: Connection, table: str, vec: Iterable[float], *, k: int = 5):
    """Return ``k`` nearest neighbours ordered by cosine distance."""
    import time

    from ..analysis.monitoring import update_metric

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT node_id, space FROM {table} ORDER BY vec <-> %s LIMIT %s",
            (_fmt_vector(vec), k),
        )
        rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    update_metric("pgvector_query_ms", elapsed_ms)
    return rows
