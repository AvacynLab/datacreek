import importlib.abc
import importlib.util
import json
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


@pytest.mark.faiss_gpu
def test_rerank_pq_gpu_matches_cpu():
    faiss = pytest.importorskip("faiss")
    if not getattr(faiss, "StandardGpuResources", None):
        pytest.skip("faiss not compiled with GPU")

    rng = np.random.default_rng(1)
    xb = rng.standard_normal((200, 32)).astype(np.float32)
    xq = xb[:1] + 0.02
    res_gpu = hybrid_ann.rerank_pq(xb, xq, k=3, gpu=True)
    res_cpu = hybrid_ann.rerank_pq(xb, xq, k=3, gpu=False)
    np.testing.assert_array_equal(res_gpu, res_cpu)


def test_search_hnsw_pq_requires_faiss(monkeypatch):
    monkeypatch.setattr(hybrid_ann, "faiss", None)
    xb = np.zeros((1, 2), dtype=np.float32)
    xq = xb.copy()
    with pytest.raises(RuntimeError):
        hybrid_ann.search_hnsw_pq(xb, xq)


def test_hybrid_ann_bench(tmp_path, monkeypatch):
    """Hybrid ANN benchmark should meet recall and latency targets."""
    spec = importlib.util.spec_from_file_location(
        "bench_hybrid_ann",
        Path(__file__).resolve().parents[1] / "scripts" / "bench_hybrid_ann.py",
    )
    bench = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(bench)

    def exact_search(xb, xq, k=10, prefetch=50):
        sims = np.linalg.norm(xb - xq[0], axis=1)
        return np.argsort(sims)[:k]

    monkeypatch.setattr(bench, "search_hnsw_pq", exact_search)
    res = bench.run_bench(n=50, d=4, k=5, queries=5)
    assert res["recall@5"] >= 0.92
    assert res["p95_ms"] < 20
    out = tmp_path / "bench.json"
    out.write_text(json.dumps(res))
    loaded = json.loads(out.read_text())
    assert loaded == res
