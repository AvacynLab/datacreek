import numpy as np
import pytest

from datacreek.analysis.compression import FractalNetPruner


class DummyModel:
    def __init__(self):
        self.w = np.array([0.05, 0.001], dtype=float)

    def named_parameters(self):
        return [("w", self.w)]


def eval_fn(model):
    # simple "perplexity" surrogate as sum of squares
    val = float(np.sum(model.w**2))
    return val


def test_fractalnet_pruner_simple():
    pruner = FractalNetPruner(lambda_=0.03)
    pruner.model = DummyModel()
    ok, perp = pruner.prune(eval_fn)
    assert ok
    assert perp <= eval_fn(pruner.model) + 1e-9
