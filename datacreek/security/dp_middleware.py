from __future__ import annotations

"""FastAPI middleware enforcing differential privacy budgets."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

import json
import logging

logger = logging.getLogger("datacreek.privacy")

from datacreek.db import SessionLocal, TenantPrivacy

from .tenant_privacy import can_consume_epsilon


class DPBudgetMiddleware(BaseHTTPMiddleware):
    """Check and decrement tenant privacy budgets on each request.

    The middleware expects headers ``X-Tenant`` (integer tenant id) and
    ``X-Epsilon`` (float amount). If either is missing the request is
    passed through unchanged. When present, ``can_consume_epsilon`` is
    called to atomically update the tenant's budget. Remaining epsilon is
    returned via ``X-Epsilon-Remaining`` on the response. If the budget
    is insufficient the request is rejected with HTTP 403 and the header
    indicates the available remainder.
    """

    header_tenant = "X-Tenant"
    header_epsilon = "X-Epsilon"

    async def dispatch(self, request: Request, call_next):
        tenant = request.headers.get(self.header_tenant)
        epsilon = request.headers.get(self.header_epsilon)
        if tenant is None or epsilon is None:
            return await call_next(request)
        try:
            tid = int(tenant)
            amount = float(epsilon)
        except ValueError:
            return Response("invalid epsilon", status_code=400)

        with SessionLocal() as db:
            if not can_consume_epsilon(db, tid, amount):
                entry = db.get(TenantPrivacy, tid)
                remaining = 0.0
                if entry is not None:
                    remaining = entry.epsilon_max - entry.epsilon_used
                logger.info(
                    json.dumps(
                        {"tenant": tid, "eps_req": amount, "allowed": False}
                    )
                )
                return Response(
                    status_code=403,
                    headers={"X-Epsilon-Remaining": f"{remaining:.6f}"},
                )
            entry = db.get(TenantPrivacy, tid)
            remaining = 0.0
            if entry is not None:
                remaining = entry.epsilon_max - entry.epsilon_used

        response = await call_next(request)
        response.headers["X-Epsilon-Remaining"] = f"{remaining:.6f}"
        logger.info(
            json.dumps({"tenant": tid, "eps_req": amount, "allowed": True})
        )
        return response
