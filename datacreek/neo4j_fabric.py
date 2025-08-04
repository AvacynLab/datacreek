"""Utilities for interacting with Neo4j Fabric in a multi-tenant setup.

This module provides a lightweight client that ensures every Cypher query is
routed to a tenant-specific database using the ``USING DATABASE`` clause. It
also exposes a helper to compute the Flyway migration path for a given tenant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # optional neo4j dependency
    from neo4j import Driver, Result  # type: ignore
except Exception:  # pragma: no cover - dependency not installed for tests
    Driver = Any  # type: ignore
    Result = Any  # type: ignore


@dataclass
class Neo4jFabricClient:
    """Client wrapper that routes queries to a tenant database in Neo4j Fabric."""

    driver: Driver

    def run(self, tenant: str, cypher: str, **params: Any) -> Result:
        """Execute a Cypher statement against ``tenant``'s database.

        Parameters
        ----------
        tenant:
            Identifier of the tenant's Fabric database.
        cypher:
            Cypher query to execute *without* a ``USING DATABASE`` clause.
        params:
            Parameters forwarded to the Neo4j driver.

        Returns
        -------
        Result
            The result returned by :mod:`neo4j`.
        """
        statement = f"USING DATABASE {tenant} {cypher}"
        # ``session`` implements the context manager protocol on the real driver
        with self.driver.session() as session:  # type: ignore[union-attr]
            return session.run(statement, **params)  # type: ignore[arg-type]


def flyway_migration_path(tenant: str) -> Path:
    """Return the Flyway migration directory for ``tenant``.

    Examples
    --------
    >>> flyway_migration_path("alpha")
    PosixPath('db/migration/alpha')
    """
    return Path("db") / "migration" / tenant
