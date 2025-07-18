from __future__ import annotations

import numpy as np

try:
    import scipy.linalg as la
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional dependency
    la = None  # type: ignore
    sp = None  # type: ignore

__all__ = ["chebyshev_diag_hutchpp"]


def chebyshev_diag_hutchpp(
    L: np.ndarray | "sp.spmatrix",
    t: float,
    *,
    order: int = 7,
    samples: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``diag(exp(-t L))`` using a Hutch++ fallback.

    When :mod:`scipy` is available the diagonal is computed exactly via
    :func:`scipy.linalg.expm`. Otherwise a Hutchinson estimator with ``samples``
    probes is used.
    """
    if la is not None:
        arr = L.toarray() if sp is not None and sp.issparse(L) else np.asarray(L)
        return np.diag(la.expm(-t * arr))

    rng = np.random.default_rng(rng)
    arr = L.toarray() if sp is not None and sp.issparse(L) else np.asarray(L)
    n = arr.shape[0]
    Z = rng.choice([-1.0, 1.0], size=(n, samples))
    Y = arr @ Z
    return np.mean(Y * Z, axis=1)
