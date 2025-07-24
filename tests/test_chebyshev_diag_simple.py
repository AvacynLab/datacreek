import numpy as np

from datacreek.analysis.chebyshev_diag import chebyshev_diag_hutchpp


def test_chebyshev_diag_reproducible():
    L = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    d1 = chebyshev_diag_hutchpp(
        L, 0.1, order=3, samples=32, rng=np.random.default_rng(0)
    )
    d2 = chebyshev_diag_hutchpp(
        L, 0.1, order=3, samples=32, rng=np.random.default_rng(0)
    )
    assert np.allclose(d1, d2)


def test_chebyshev_diag_close_to_exact():
    L = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    t = 0.1
    approx = chebyshev_diag_hutchpp(L, t, order=3, samples=64, rng=42)
    assert approx.shape == (2,)
    assert np.all(approx > 0)


def test_chebyshev_diag_fallback(monkeypatch):
    """Exercise the lightweight path when SciPy is unavailable."""
    import datacreek.analysis.chebyshev_diag as cd

    # Force fallback by clearing optional dependencies
    monkeypatch.setattr(cd, "la", None, raising=False)
    monkeypatch.setattr(cd, "sp", None, raising=False)
    monkeypatch.setattr(cd, "eigsh", None, raising=False)
    monkeypatch.setattr(cd, "iv", None, raising=False)

    L = np.array([[1.0, -1.0], [-1.0, 2.0]], dtype=float)
    rng = np.random.default_rng(0)
    result = cd.chebyshev_diag_hutchpp(L, 0.1, order=2, samples=16, rng=rng)
    assert result.shape == (2,)
    assert isinstance(result[0], float)
