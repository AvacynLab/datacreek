try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
import importlib.abc
import importlib.util
from pathlib import Path

import pytest

if np is not None:
    spec = importlib.util.spec_from_file_location(
        "poincare_recentering",
        Path(__file__).resolve().parents[1]
        / "datacreek"
        / "analysis"
        / "poincare_recentering.py",
    )
    poincare_recentering = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(poincare_recentering)
    recenter_embeddings = poincare_recentering.recenter_embeddings


def test_recenter_embeddings_returns_origin_mean():
    if np is None:
        pytest.skip("numpy not installed")
    embs = {0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])}
    rec = recenter_embeddings(embs)
    vals = np.stack(list(rec.values()))
    center = vals.mean(axis=0)
    assert np.linalg.norm(center) < 1e-3
    assert all(np.linalg.norm(v) < 1 - 1e-6 for v in vals)
