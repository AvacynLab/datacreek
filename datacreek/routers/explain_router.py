from __future__ import annotations

"""Explainability router for public graph exploration.

This module exposes a simple endpoint returning a subgraph and attention
scores around a given node as well as a base64 encoded SVG visualisation.
The route requires authentication via the ``X-API-Key`` header so it can
be used from demo pages.
"""

import base64

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from datacreek.analysis import explain_to_svg
from datacreek.backends import get_neo4j_driver, get_redis_client
from datacreek.core.dataset import DatasetBuilder
from datacreek.db import SessionLocal, User
from datacreek.services import get_user_by_key

router = APIRouter(prefix="/explain", tags=["explain"])


def get_current_user(api_key: str = Header(..., alias="X-API-Key")) -> User:
    """Authenticate using the API key stored in the database or Redis."""
    with SessionLocal() as db:
        user = get_user_by_key(db, api_key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user


def _load_dataset(name: str, user: User) -> DatasetBuilder:
    """Return ``DatasetBuilder`` for ``name`` if accessible by ``user``."""
    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    finally:
        if driver:
            driver.close()
    if ds.owner_id not in {None, user.id}:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds.redis_client = client
    return ds


@router.get("/{node}", summary="Explain node neighborhood")
def explain_node_public(
    node: str,
    dataset: str = Query(
        "demo",
        alias="dataset",
        example="demo",
    ),
    hops: int = Query(3, ge=1, le=5),
    user: User = Depends(get_current_user),
) -> JSONResponse:
    """Return explanation data for ``node`` in ``dataset``.

    The response contains the neighborhood ``nodes`` and ``edges`` lists,
    an ``attn`` mapping for attention scores and a base64 encoded ``svg``
    rendering of the subgraph.

    Example
    -------

    ``curl``::

        curl -H "X-API-Key: <token>" \
             "http://localhost:8000/explain/foo?dataset=demo"

    JavaScript ``fetch``::

        fetch("/explain/foo?dataset=demo", {
            headers: {"X-API-Key": "<token>"}
        }).then(r => r.json());
    """

    ds = _load_dataset(dataset, user)
    data = ds.explain_node(node, hops=hops)
    svg = explain_to_svg(data)
    payload = {
        "nodes": data.get("nodes", []),
        "edges": data.get("edges", []),
        "attn": data.get("attention", {}),
        "svg": base64.b64encode(svg.encode()).decode(),
    }
    return JSONResponse(payload)
