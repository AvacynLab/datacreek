import argparse
import importlib.abc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "hybrid_ann",
    Path(__file__).resolve().parents[1] / "datacreek" / "analysis" / "hybrid_ann.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert isinstance(SPEC.loader, importlib.abc.Loader)
SPEC.loader.exec_module(MODULE)  # type: ignore
search_hnsw_pq = MODULE.search_hnsw_pq


def run_bench(n: int = 5000, d: int = 32, k: int = 100, queries: int = 100) -> dict:
    """Benchmark Hybrid ANN recall and latency."""
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    xq = xb[:queries] + 0.01

    gt_dist = np.linalg.norm(xb[None, :] - xq[:, None, :], axis=2)
    gt = np.argsort(gt_dist, axis=1)[:, :k]

    times = []
    recall = []
    for i in range(queries):
        t0 = time.perf_counter()
        idx = search_hnsw_pq(xb, xq[i : i + 1], k=k, prefetch=500)
        times.append(time.perf_counter() - t0)
        recall.append(len(set(idx) & set(gt[i])) / k)

    return {
        f"recall@{k}": float(np.mean(recall)),
        "p95_ms": float(np.percentile(np.array(times) * 1000, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_hybrid_ann.json")
    args = parser.parse_args()
    res = run_bench()
    with open(args.output, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
