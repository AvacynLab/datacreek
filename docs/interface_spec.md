# Stable Interface Specification

This document enumerates the APIs considered stable for Datacreek v1.1. Any
future optimisation must preserve their behaviour and signature.

## Topology Pipeline
- `tpl_incremental(graph: nx.Graph, radius: int = 1, dimension: int = 1) -> dict`
  - Updates the persistence diagrams around each node if its neighbourhood hash
    changed. Returns a mapping of node IDs to diagrams.

## Chebyshev GraphWave
- `chebyshev_diag_hutchpp(L: array, t: float, order: int = 7, samples: int = 64, rng=None) -> ndarray`
  - Estimates `diag(exp(-t L))` using Hutch++ when SciPy is unavailable.
  - For sparse Laplacians, matrix-vector products may be offloaded to cuSPARSE.

## Autotune + Metrics
- `autotune_node2vec(kg, queries, ground_truth, k=10, var_threshold=1e-4, max_evals=40)`
  - Bayesian optimisation of Node2Vec parameters `p` and `q`.
- `autotune_nprobe(index, queries, truths, recall_goal=0.92)`
  - Tunes the FAISS `nprobe` parameter for IVFPQ indices using recall@100.

## Governance CI
- Continuous integration runs `bench_all.py` and fails when performance
  regresses by more than 5 % compared to `benchmarks/baseline.json`.

