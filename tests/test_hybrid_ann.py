import importlib.abc
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "datacreek.analysis.hybrid_ann",
    Path(__file__).resolve().parents[1] / "datacreek" / "analysis" / "hybrid_ann.py",
)
hybrid_ann = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(hybrid_ann)


@pytest.mark.faiss_gpu
def test_search_hnsw_pq_pipeline():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((100, 16)).astype(np.float32)
    xq = xb[:1] + 0.01
    res = hybrid_ann.search_hnsw_pq(xb, xq, k=5, prefetch=20)
    assert len(res) == 5
    assert all(0 <= i < 100 for i in res)


def test_search_hnsw_pq_requires_faiss(monkeypatch):
    monkeypatch.setattr(hybrid_ann, "faiss", None)
    xb = np.zeros((1, 2), dtype=np.float32)
    xq = xb.copy()
    with pytest.raises(RuntimeError):
        hybrid_ann.search_hnsw_pq(xb, xq)
