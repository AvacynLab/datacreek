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

if np is not None and poincare_recentering.torch is not None:
    torch = poincare_recentering.torch
else:  # pragma: no cover - torch optional
    torch = None


def test_recenter_embeddings_returns_origin_mean():
    if np is None:
        pytest.skip("numpy not installed")
    embs = {0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])}
    rec = recenter_embeddings(embs)
    vals = np.stack(list(rec.values()))
    center = vals.mean(axis=0)
    assert np.linalg.norm(center) < 1e-3
    assert all(np.linalg.norm(v) < 1 - 1e-6 for v in vals)


def test_exp_log_inverse():
    if np is None:
        pytest.skip("numpy not installed")
    x = np.array([0.2, -0.1])
    v = poincare_recentering._log_map(np.zeros(2), x)
    y = poincare_recentering._exp_map_zero(v)
    assert np.linalg.norm(y - x) < 1e-6


def test_exp_log_grad_fp16():
    if torch is None:
        pytest.skip("torch not installed")
    x = torch.tensor([0.2, 0.1], dtype=torch.float16, requires_grad=True)
    v = poincare_recentering._log_map_zero_torch(x)
    y = poincare_recentering._exp_map_zero_torch(v)
    jac = torch.autograd.functional.jacobian(
        lambda z: poincare_recentering._exp_map_zero_torch(
            poincare_recentering._log_map_zero_torch(z)
        ),
        x,
    )
    assert torch.allclose(jac, torch.eye(2, dtype=torch.float16), rtol=1e-3, atol=1e-3)


def test_measure_overshoot_reduces_with_clamp():
    if np is None:
        pytest.skip("numpy not installed")
    radii = [0.5, 1.0, 1.5]
    res = poincare_recentering.measure_overshoot(radii, kappa=-1.0, num_samples=32)
    assert all(a <= b for a, b in zip(res["clamp"], res["noclamp"]))
