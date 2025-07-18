"""Tenant-level differential privacy budgets."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..db import TenantPrivacy


def set_tenant_limit(db: Session, tenant_id: int, epsilon_max: float) -> None:
    """Create or update the privacy budget for ``tenant_id``."""
    entry = db.get(TenantPrivacy, tenant_id)
    if entry is None:
        entry = TenantPrivacy(tenant_id=tenant_id, epsilon_max=epsilon_max)
        db.add(entry)
    else:
        entry.epsilon_max = epsilon_max
    db.commit()


def can_consume_epsilon(db: Session, tenant_id: int, amount: float) -> bool:
    """Return True and update usage if ``amount`` fits the remaining budget."""
    entry = db.get(TenantPrivacy, tenant_id)
    if entry is None:
        return False
    if entry.epsilon_used + amount > entry.epsilon_max:
        return False
    entry.epsilon_used += amount
    db.commit()
    return True
