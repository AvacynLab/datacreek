#!/usr/bin/env python3
"""Aggregate benchmark results and system metrics."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") in sys.path:
    sys.path.remove(str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import argparse
import importlib
import json
import time

psutil_spec = importlib.util.find_spec("psutil")
if psutil_spec:
    import psutil  # type: ignore
else:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

np = None
if importlib.util.find_spec("numpy") is not None:
    import numpy as np  # type: ignore


def check_regression(
    metrics: dict[str, float], baseline: dict[str, float], threshold: float = 0.05
) -> None:
    """Raise SystemExit if metrics regress beyond threshold.

    Latency, CPU and memory must not exceed baseline by more than
    ``threshold``. Recall must not drop below the baseline minus
    ``threshold``.
    """
    regressions = []
    if metrics["p95_latency"] > baseline["p95_latency"] * (1 + threshold):
        regressions.append("p95_latency")
    if metrics["cpu_percent"] > baseline["cpu_percent"] * (1 + threshold):
        regressions.append("cpu_percent")
    if metrics["memory_mb"] > baseline["memory_mb"] * (1 + threshold):
        regressions.append("memory_mb")
    if metrics["recall"] < baseline["recall"] * (1 - threshold):
        regressions.append("recall")
    if regressions:
        reg = ", ".join(regressions)
        raise SystemExit(f"Benchmark regression detected in: {reg}")


def benchmark(
    backend: str = "flat", n: int = 1000, d: int = 128, k: int = 5
) -> tuple[float, float]:
    """Return (p95 latency, recall@k) for a random graph."""
    if np is None:
        raise SystemExit("bench_all requires numpy")
    from datacreek.core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    for i in range(n):
        kg.graph.add_node(f"n{i}", embedding=np.random.rand(d).tolist())

    kg.build_faiss_index(method=backend)

    q = np.random.rand(d)
    latencies: list[float] = []
    recall_hits = 0
    vectors = np.vstack([kg.graph.nodes[n]["embedding"] for n in kg.graph.nodes])
    base = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    true_top = set(np.argsort(base)[-k:])

    for _ in range(100):
        start = time.monotonic()
        res = kg.search_faiss(q, k=k)
        latencies.append(time.monotonic() - start)
        idxs = [int(r[1:]) for r in res]
        recall_hits += len(true_top.intersection(idxs))

    p95 = float(np.quantile(latencies, 0.95))
    recall = recall_hits / (100 * k)
    return p95, recall


def gpu_count() -> int:
    """Return available GPU count if FAISS supports it."""
    try:
        import faiss

        return faiss.get_num_gpus()
    except Exception:  # pragma: no cover - optional dependency
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="flat")
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    if psutil is None:
        raise SystemExit("bench_all requires psutil")
    proc = psutil.Process()
    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss / 1e6
    p95, recall = benchmark(args.backend)
    cpu_after = proc.cpu_percent(interval=None)
    mem_after = proc.memory_info().rss / 1e6
    metrics = {
        "cpu_percent": cpu_after - cpu_before,
        "memory_mb": mem_after - mem_before,
        "gpu_count": gpu_count(),
        "p95_latency": p95,
        "recall": recall,
    }
    print(json.dumps(metrics))
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        check_regression(metrics, baseline, args.threshold)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
