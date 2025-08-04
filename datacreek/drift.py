"""Utilities for detecting embedding drift and scheduling retraining.

This module exposes small helpers to compute a drift *level* given a
numeric score and decide whether retraining should be triggered.  The
thresholds are derived from the SaaS readiness checklist:

* warn when drift exceeds 0.07
* trigger critical alert and retraining when drift exceeds 0.10

The functions are intentionally tiny so they can be reused both in Airflow
DAGs and in unit tests without pulling heavy dependencies such as NumPy.
"""

from __future__ import annotations

WARN_THRESHOLD: float = 0.07
"""Drift value above which the system should emit a warning."""

CRIT_THRESHOLD: float = 0.10
"""Drift value above which the system must trigger retraining."""


def drift_level(value: float) -> str:
    """Classify the drift *value* into qualitative buckets.

    Parameters
    ----------
    value:
        Measured drift magnitude, typically a distance between
        recent and reference embedding distributions.

    Returns
    -------
    str
        One of ``"ok"``, ``"warn"`` or ``"crit"`` according to the
        thresholds defined in this module.
    """

    if value >= CRIT_THRESHOLD:
        return "crit"
    if value >= WARN_THRESHOLD:
        return "warn"
    return "ok"


def should_trigger_retrain(value: float) -> bool:
    """Return ``True`` when the *value* requires starting retraining.

    This is a convenience wrapper around :func:`drift_level` that makes the
    intent explicit when used by orchestration code.
    """

    return value >= CRIT_THRESHOLD
