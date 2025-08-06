import json

import numpy as np

from datacreek.hypergraph import (
    EDGE_DOC,
    EDGE_TAG,
    RedisGraphHotLayer,
    macro_f1,
    optimise_alpha,
)


def test_add_edge_records_type():
    hot = RedisGraphHotLayer()
    hot.add_edge("u", "d", EDGE_DOC)
    hot.add_edge("u", "t", EDGE_TAG)
    assert set(hot.neighbours("u")) == {"d", "t"}
    assert set(hot.neighbours("u", EDGE_DOC)) == {"d"}
    assert set(hot.neighbours("u", EDGE_TAG)) == {"t"}


def test_optimise_alpha_improves_f1_and_persists(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    preds1 = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
        ]
    )
    preds2 = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )
    baseline_probs = 0.5 * preds1 + 0.5 * preds2
    baseline_f1 = macro_f1(y_true, baseline_probs.argmax(axis=1))

    alpha = optimise_alpha(
        [preds1, preds2], y_true, steps=50, save_path=tmp_path / "alpha.json"
    )
    assert np.isclose(alpha.sum(), 1.0, atol=1e-4)

    opt_probs = alpha[0] * preds1 + alpha[1] * preds2
    opt_f1 = macro_f1(y_true, opt_probs.argmax(axis=1))
    assert opt_f1 >= baseline_f1 + 0.01

    with open(tmp_path / "alpha.json") as fh:
        stored = np.array(json.load(fh))
    assert np.allclose(stored, alpha)
