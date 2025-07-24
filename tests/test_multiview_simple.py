"""Lightweight tests for multiview helpers.

These functions normally rely on scikit-learn, but other tests stub that
library. Here we make sure to reload the real modules so the algorithms run
correctly.
"""

import importlib
import sys
import pytest

import numpy as np
import pytest

# Remove any sklearn stubs inserted by other tests and reload the helpers
for mod in [
    "sklearn",
    "sklearn.cross_decomposition",
    "sklearn.decomposition",
    "sklearn.linear_model",
    "sklearn.gaussian_process",
    "scipy",
    "scipy.stats",
]:
    sys.modules.pop(mod, None)
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
import sklearn.cross_decomposition  # noqa: E402

import datacreek.analysis.multiview as multiview
importlib.reload(multiview)

from datacreek.analysis.multiview import (
    product_embedding,
    train_product_manifold,
    aligned_cca,
    cca_align,
    load_cca,
    hybrid_score,
    multiview_contrastive_loss,
    meta_autoencoder,
)


def test_product_embedding_and_train():
    hyper = {"a": [0.0, 0.5], "b": [0.5, 0.0]}
    eucl = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    emb = product_embedding(hyper, eucl)
    assert set(emb) == {"a", "b"}
    assert np.allclose(emb["a"], [0.0, 0.5, 1.0, 0.0])

    def distance(h, e):
        return np.linalg.norm(h["a"] - h["b"]) + np.linalg.norm(e["a"] - e["b"])

    before = distance({k: np.asarray(v) for k, v in hyper.items()}, {k: np.asarray(v) for k, v in eucl.items()})
    h_new, e_new = train_product_manifold(hyper, eucl, [("a", "b")], epochs=5, lr=0.1)
    after = distance(h_new, e_new)
    assert after < before


def test_aligned_cca_and_load(tmp_path, caplog):
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [0.5, 0.5], "b": [-0.5, -0.5]}
    path = tmp_path / "cca.pkl"
    caplog.set_level("INFO")
    latent = cca_align(n2v, gw, n_components=1, path=str(path))
    assert path.exists()
    assert set(latent) == {"a", "b"}
    assert any("cca_sha=" in r.message for r in caplog.records)

    W1, W2 = load_cca(str(path))
    assert W1.shape[1] == 1
    assert not W1.flags.writeable
    assert W2.shape == W1.shape


def test_aligned_cca_empty():
    latent, model = aligned_cca({}, {})
    assert latent == {}
    assert hasattr(model, "fit")


def test_hybrid_and_losses():
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [0.9, 0.1], "b": [0.1, 0.9]}
    hyp = {"a": [0.1, 0.0], "b": [0.0, 0.1]}

    score = hybrid_score(n2v["a"], n2v["b"], gw["a"], gw["b"], hyp["a"], hyp["b"])
    assert 0.0 <= score <= 1.0

    loss = multiview_contrastive_loss(n2v, gw, hyp, tau=0.5)
    assert loss > 0.0

    latent, recon = meta_autoencoder(n2v, gw, hyp, bottleneck=1)
    assert set(latent) == {"a", "b"}
    assert recon["a"].shape[0] == 6
