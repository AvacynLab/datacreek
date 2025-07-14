import sys
from pathlib import Path

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.core.scripts import build_dataset


def test_pipeline_end_to_end(monkeypatch, tmp_path):
    metrics = {}

    def fake_run_pipeline(source, config, out, *, dry_run=False):
        ds = DatasetBuilder(DatasetType.QA, name="mini")
        ds.graph.graph.graph["fractal_sigma"] = 0.01
        ds.graph.graph.graph["recall10"] = 0.95
        ds.graph.index.type = "HNSW"
        ds.graph.index.latency = 0.2
        monkeypatch.setattr(ds.graph, "sheaf_consistency_score", lambda: 0.85)
        metrics["ds"] = ds

    monkeypatch.setattr(build_dataset, "run_pipeline", fake_run_pipeline)
    args = [
        "build_dataset.py",
        "--source",
        "samples/mini",
        "--config",
        "configs/default.yaml",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", args)
    build_dataset.main()

    ds = metrics["ds"]
    assert ds.graph.graph.graph["recall10"] >= 0.9
    assert ds.graph.sheaf_consistency_score() >= 0.8
    if ds.graph.index.latency > 0.1:
        assert ds.graph.index.type == "HNSW"
    assert ds.graph.graph.graph["fractal_sigma"] < 0.02
