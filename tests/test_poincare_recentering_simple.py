import types
import sys
import numpy as np
import pytest

from datacreek.analysis import poincare_recentering as pr


def test_mobius_geo_roundtrip():
    v = np.array([0.2, 0.05])
    y = pr._exp_map_zero(v)
    recovered = pr._log_map(np.zeros(2), y)
    assert np.allclose(recovered, v, atol=1e-6)
    zero = pr._mobius_add(y, pr._mobius_neg(y), clamp=False)
    assert np.allclose(zero, np.zeros(2), atol=1e-12)


def test_clamp_and_radius():
    x = np.array([0.8, 0.8])
    clamped = pr._clamp_ball(x.copy())
    assert np.linalg.norm(clamped) < 1
    rad = pr.hyperbolic_radius(clamped)
    assert rad > 0


def test_measure_overshoot_zero():
    res = pr.measure_overshoot([0.1, 0.2], num_samples=5)
    assert res == {"clamp": [0.0, 0.0], "noclamp": [0.0, 0.0]}


def test_recenter_embeddings_symmetry():
    emb = {0: [0.2, 0.0], 1: [-0.2, 0.0]}
    out = pr.recenter_embeddings(emb)
    out_arr = np.vstack(list(out.values()))
    orig_arr = np.vstack(list(emb.values()))
    assert np.allclose(out_arr, orig_arr, atol=1e-3)


def test_recenter_embeddings_shift():
    emb = {0: [0.1, 0.0], 1: [0.3, 0.0]}
    out = pr.recenter_embeddings(emb)
    arr = np.stack(list(out.values()))
    assert np.allclose(arr.mean(axis=0), 0, atol=1e-2)


def test_trace_overshoot_parquet(monkeypatch, tmp_path):
    written = {}

    class DummyPQ:
        def write_table(self, table, path):
            written['table'] = table
            written['path'] = path

    dummy_pq = DummyPQ()

    def dummy_table(data):
        written['data'] = data
        return data

    fake_pa = types.SimpleNamespace(table=dummy_table, parquet=dummy_pq)
    monkeypatch.setitem(sys.modules, 'pyarrow', fake_pa)
    monkeypatch.setitem(sys.modules, 'pyarrow.parquet', dummy_pq)

    out_path = tmp_path / 'res.parquet'
    pr.trace_overshoot_parquet(str(out_path), num_points=2, curvatures=(-1.0,))

    assert written['path'] == str(out_path)
    assert 'kappa' in written['data']
    assert len(written['data']['kappa']) == 2
