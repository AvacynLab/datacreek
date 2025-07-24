import sys
import types

import numpy as np

import datacreek.analysis.compression as comp


def test_save_restore(tmp_path):
    file = tmp_path / "model.pkl"
    data = {"x": 1}
    comp.save_checkpoint(str(file), data)
    assert comp.restore_checkpoint(str(file)) == data


def test_prune_fractalnet_list(monkeypatch):
    monkeypatch.setattr(comp, "np", None)
    weights = [1.0, -2.0, 3.0, -4.0]
    pruned = comp.prune_fractalnet(weights, ratio=0.5)
    assert sum(1 for x in pruned if x) == 2


def test_prune_fractalnet_np(monkeypatch):
    monkeypatch.setattr(comp, "np", np)
    weights = np.array([1.0, -2.0, 3.0, -4.0])
    pruned = comp.prune_fractalnet(weights, ratio=0.5)
    assert pruned.shape == weights.shape
    assert np.count_nonzero(pruned) == 2


class DummyModel:
    def __init__(self, arr):
        self.weight = np.array(arr, dtype=float)

    def named_parameters(self):
        return [("weight", self.weight)]


def test_fractalnet_pruner(monkeypatch):
    pruner = comp.FractalNetPruner(lambda_=0.5)
    pruner.model = DummyModel([1.0, -1.0, 0.5])

    stub_config = types.SimpleNamespace(load_config=lambda: {})
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", stub_config)
    monkeypatch.setattr(comp, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(comp, "restore_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(comp, "prune_reverts_total", None)

    accepted, ppl = pruner.prune(lambda m: 1.0, baseline=1.0)
    assert accepted
    assert ppl == 1.0


def test_fractalnet_pruner_revert(monkeypatch):
    pruner = comp.FractalNetPruner(lambda_=0.5)
    pruner.model = DummyModel([1.0, -1.0, 0.5])

    stub_config = types.SimpleNamespace(load_config=lambda: {})
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", stub_config)
    called = {}

    def fake_restore(path="fractal.bak"):
        called["called"] = True
        return pruner.model

    monkeypatch.setattr(comp, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(comp, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(comp, "prune_reverts_total", None)

    accepted, ppl = pruner.prune(lambda m: 1.1, baseline=1.0)
    assert not accepted
    assert ppl == 1.1
    assert called.get("called")
