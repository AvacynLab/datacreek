from __future__ import annotations

import os
from typing import Optional

import redis
from neo4j import Driver, GraphDatabase

from datacreek.core.dataset import DatasetBuilder
from datacreek.utils.config import get_neo4j_config, get_redis_config, load_config

_config = load_config()


def get_neo4j_driver(cfg: Optional[dict] = None) -> Driver:
    """Return a Neo4j driver using config or environment variables."""
    cfg = get_neo4j_config(cfg or _config)
    uri = os.getenv("NEO4J_URI", cfg.get("uri", "bolt://localhost:7687"))
    user = os.getenv("NEO4J_USER", cfg.get("user", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD", cfg.get("password", "neo4j"))
    return GraphDatabase.driver(uri, auth=(user, password))


def get_redis_client(cfg: Optional[dict] = None) -> redis.Redis:
    """Return a Redis client using config or environment variables."""
    cfg = get_redis_config(cfg or _config)
    host = os.getenv("REDIS_HOST", cfg.get("host", "localhost"))
    port = int(os.getenv("REDIS_PORT", cfg.get("port", 6379)))
    return redis.Redis(host=host, port=port, decode_responses=True)


def persist_dataset(ds: DatasetBuilder) -> None:
    """Persist *ds* to Redis and Neo4j using default connections."""
    client = get_redis_client()
    driver = get_neo4j_driver()
    ds.save_state(client, driver, redis_key=f"dataset:{ds.name}")
    driver.close()
    try:
        client.close()
    except Exception:
        pass


def load_dataset(name: str) -> DatasetBuilder | None:
    """Load dataset ``name`` from Redis/Neo4j if it exists."""
    client = get_redis_client()
    driver = get_neo4j_driver()
    try:
        ds = DatasetBuilder.load_state(
            redis_client=client,
            neo4j_driver=driver,
            redis_key=f"dataset:{name}",
        )
    except Exception:
        ds = None
    driver.close()
    try:
        client.close()
    except Exception:
        pass
    return ds


def delete_dataset(name: str) -> None:
    """Remove dataset ``name`` from Redis and Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()

    client.delete(f"dataset:{name}")

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    driver.close()
    try:
        client.close()
    except Exception:
        pass
