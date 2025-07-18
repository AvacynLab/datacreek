import importlib.util
import json
import os
import subprocess
import sys

import pytest

from scripts.bench_all import check_regression


def test_bench_all_script(tmp_path):
    if any(
        importlib.util.find_spec(pkg) is None for pkg in ("pydantic", "numpy", "psutil")
    ):
        pytest.skip("missing heavy dependencies")
    import site

    env = {"PYTHONPATH": os.pathsep.join(site.getsitepackages() + ["."])}
    out = subprocess.check_output(
        [sys.executable, "scripts/bench_all.py", "--backend", "flat"],
        env=env,
    )
    metrics = json.loads(out)
    assert {
        "cpu_percent",
        "memory_mb",
        "gpu_count",
        "p95_latency",
        "recall",
    } <= metrics.keys()
    assert 0 <= metrics["recall"] <= 1
    assert metrics["p95_latency"] > 0


def test_check_regression():
    metrics = {
        "cpu_percent": 10,
        "memory_mb": 20,
        "gpu_count": 0,
        "p95_latency": 0.2,
        "recall": 0.9,
    }
    baseline = {
        "cpu_percent": 10,
        "memory_mb": 20,
        "gpu_count": 0,
        "p95_latency": 0.21,
        "recall": 0.88,
    }
    check_regression(metrics, baseline)
    baseline["recall"] = 0.95
    with pytest.raises(SystemExit):
        check_regression(metrics, baseline)
