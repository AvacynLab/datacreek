import numpy as np
import pytest

import datacreek.analysis.hybrid_ann as han


def test_expected_recall_and_nprobe_multi():
    # basic sanity checks for pure helpers
    recall = han.expected_recall(10, 100, 2)
    assert recall == pytest.approx(1 - (1 - 0.1) ** 2)
    assert han.choose_nprobe_multi(256, 16) == 4


def test_functions_require_faiss(monkeypatch):
    monkeypatch.setattr(han, "faiss", None)
    with pytest.raises(RuntimeError):
        han.load_ivfpq_cpu("dummy", 1)
    xb = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError):
        han.rerank_pq(xb, xb)
    with pytest.raises(RuntimeError):
        han.search_hnsw_pq(xb, xb)
