import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

# Older versions of scikit-optimize expect ``np.int`` which was removed in
# NumPy 2.x. Provide an alias so the optimizer functions work across
# versions without modification.
if not hasattr(np, "int"):
    np.int = int

sys.modules.pop("skopt", None)

import datacreek.analysis.nprobe_tuning as nt


class FakeIndexFlatIP:
    """Simple index computing inner product search."""

    def __init__(self, dim: int):
        self.dim = dim
        self.xb: np.ndarray | None = None

    def add(self, xb: np.ndarray) -> None:
        self.xb = xb

    def search(self, xq: np.ndarray, k: int):
        scores = xq @ self.xb.T
        order = np.argsort(-scores, axis=1)
        return None, order[:, :k]


class FakeIndexIVFPQ(FakeIndexFlatIP):
    """Minimal IVFPQ stub with ``nprobe`` attribute."""

    def __init__(self, quantizer: FakeIndexFlatIP, dim: int, *_args):
        super().__init__(dim)
        self.quantizer = quantizer
        self.nprobe = 32

    def train(self, xb: np.ndarray) -> None:
        self.quantizer.add(xb)


fake_faiss = SimpleNamespace(IndexFlatIP=FakeIndexFlatIP, IndexIVFPQ=FakeIndexIVFPQ)


def _build_index(xb: np.ndarray) -> FakeIndexIVFPQ:
    d = xb.shape[1]
    quantizer = FakeIndexFlatIP(d)
    index = FakeIndexIVFPQ(quantizer, d, 1, 1, 4)
    index.train(xb)
    index.add(xb)
    return index


def test_compute_recall_simple():
    result = np.array([[1, 2, 3], [0, 4, 5]])
    ground = [[0, 2, 9], [4, 5, 6]]
    assert nt._compute_recall(result, ground) == 0.5


def test_autotune_nprobe_basic(monkeypatch):
    d = 4
    rng = np.random.default_rng(0)
    xb = rng.random((40, d), dtype=np.float32)
    xq = rng.random((5, d), dtype=np.float32)
    index = _build_index(xb)
    monkeypatch.setattr(nt, "faiss", fake_faiss)
    class DummyOpt:
        def __init__(self, *a, **k):
            self.calls = 0
            self.values = [32, 64, 96]
        def ask(self):
            val = self.values[self.calls % len(self.values)]
            self.calls += 1
            return [val]
        def tell(self, *a, **k):
            pass

    monkeypatch.setattr(nt, "Optimizer", DummyOpt)
    best = nt.autotune_nprobe(index, xb, xq, k=5, target=0.0, max_evals=3)
    assert 32 <= best <= 256
    assert index.nprobe == best


def test_profile_nprobe_cycle(monkeypatch, tmp_path):
    d = 4
    rng = np.random.default_rng(1)
    xb = rng.random((20, d), dtype=np.float32)
    xq = rng.random((4, d), dtype=np.float32)
    index = _build_index(xb)
    monkeypatch.setattr(nt, "faiss", fake_faiss)
    times = iter([0.0, 0.1, 0.1, 0.2])
    monkeypatch.setattr(nt.time, "monotonic", lambda: next(times))
    path = tmp_path / "prof.pkl"
    out = nt.profile_nprobe(index, xb, xq, k=5, nprobes=[32, 64], path=path)
    assert path.exists()
    assert out["nprobe"] == [32, 64]
    assert len(out["latency"]) == 2 and len(out["recall"]) == 2


def test_nprobe_no_faiss(monkeypatch):
    monkeypatch.setattr(nt, "faiss", None)
    with pytest.raises(RuntimeError):
        nt.autotune_nprobe(None, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(RuntimeError):
        nt.profile_nprobe(None, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))
