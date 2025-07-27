import numpy as np
import datacreek.analysis.chebyshev_diag as cd


def test_hutchpp_fallback_matches_diag(monkeypatch):
    """Fallback estimator should approximate the diagonal of L."""
    for name in ("la", "sp", "eigsh", "iv"):
        monkeypatch.setattr(cd, name, None)
    rng = np.random.default_rng(0)
    L = np.array([[2.0, -1.0], [-1.0, 4.0]])
    est = cd.chebyshev_diag_hutchpp(L, t=0.3, samples=5000, rng=rng)
    assert np.allclose(est, np.diag(L), atol=0.05)


def test_hutchpp_deterministic_seed_and_shape(monkeypatch):
    """Using the same seed returns identical results regardless of ``t``."""
    for name in ("la", "sp", "eigsh", "iv"):
        monkeypatch.setattr(cd, name, None)
    L = np.arange(9.0).reshape(3, 3)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    res1 = cd.chebyshev_diag_hutchpp(L, t=1.0, samples=100, rng=rng1)
    res2 = cd.chebyshev_diag_hutchpp(L, t=5.0, samples=100, rng=rng2)
    assert np.allclose(res1, res2)
    assert res1.shape == (3,)
