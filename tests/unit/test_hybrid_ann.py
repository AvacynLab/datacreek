import numpy as np
import pytest

from datacreek.analysis import hybrid_ann


pytestmark = pytest.mark.skipif(
    hybrid_ann.faiss is not None,
    reason="faiss installed",
)


def test_expected_recall_basic():
    assert hybrid_ann.expected_recall(8, 32, 2) == pytest.approx(0.4375)


def test_expected_recall_zero_cells():
    assert hybrid_ann.expected_recall(5, 0, 4) == 0.0


def test_choose_nprobe_multi_heuristic():
    assert hybrid_ann.choose_nprobe_multi(256, 16) == 4


def test_choose_nprobe_multi_edge():
    assert hybrid_ann.choose_nprobe_multi(0, 8) == 1


def test_faiss_required_errors():
    xb = np.zeros((10, 4), dtype=np.float32)
    xq = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(RuntimeError):
        hybrid_ann.load_ivfpq_cpu("dummy.idx", 2)
    with pytest.raises(RuntimeError):
        hybrid_ann.rerank_pq(xb, xq)
    with pytest.raises(RuntimeError):
        hybrid_ann.search_hnsw_pq(xb, xq)
