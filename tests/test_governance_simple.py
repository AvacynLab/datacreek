import numpy as np
import pytest

from datacreek.analysis import governance as gov


def test_alignment_correlation_basic():
    emb1 = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    emb2 = {"a": [1.0, 1.0], "b": [-1.0, -1.0]}
    corr = gov.alignment_correlation(emb1, emb2)
    assert -1.0 <= corr <= 1.0


def test_alignment_correlation_no_common():
    with pytest.raises(ValueError):
        gov.alignment_correlation({"a": [1.0]}, {"b": [2.0]})


def test_average_hyperbolic_radius():
    emb = {"a": [0.2, 0.0], "b": [0.0, 0.2]}
    radius = gov.average_hyperbolic_radius(emb)
    assert radius > 0.0


def test_scale_bias_wasserstein_and_metrics():
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 1.0], "b": [-1.0, -1.0]}
    hyp = {"a": [0.2, 0.0], "b": [0.0, 0.2]}
    bias = gov.scale_bias_wasserstein(n2v, gw, hyp)
    metrics = gov.governance_metrics(n2v, gw, hyp)
    assert bias == metrics["bias_wasserstein"]
    assert set(metrics) == {"alignment_corr", "hyperbolic_radius", "bias_wasserstein"}


def test_mitigate_bias_wasserstein():
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    groups = {"a": "g1", "b": "g2", "c": "g2"}
    res = gov.mitigate_bias_wasserstein(emb, groups)
    g1_mean = np.linalg.norm(res["a"])
    g2_mean = np.mean([np.linalg.norm(res[n]) for n in ["b", "c"]])
    assert abs(g1_mean - g2_mean) < 1e-6
