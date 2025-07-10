# Agent Tasks Checklist

## Pending Tasks
- [ ] 1. Implement ingestion via unstructured partitioning and chunking utilities
- [ ] 2. Encode hypergraph and minimal sheaf structure
- [ ] 3. Add GDS cleaning pipeline (wcc, triangleCount, nodeSimilarity, linkprediction)
- [ ] 4. Implement fractalization (box-covering, GPU heuristic)
- [ ] 5. Compute multi-geometry embeddings (Node2Vec, GraphWave, Poincaré)
- [ ] 6. Integrate Topological Perception Layer with GUDHI diagrams and GraphRNN
- [ ] 7. Add information-guided cleanup (entropy, MDL, Information Bottleneck)
- [ ] 8. Implement autotuning algorithm with Bayesian optimisation
- [ ] 9. Enable LLM generation with validation through HNSW and Self-Instruct
- [ ] 10. Provide compression via FractalNet and inverse Mapper
- [ ] 11. Ensure export JSONL with fractal metadata and encryption
- [ ] 12. Dashboard invariants pushed to Grafana/Slack

## Info
This project is a SaaS pipeline for dataset generation. It ingests multimodal documents, constructs a fractal hypergraph, enforces topological coherence, and uses an autotuning loop to balance entropy, bottleneck distance, IB loss and MDL. Generation is guided by LLMs and data is exported with traceability.

## History
- commit 46f1749 added autotune step, crypto helpers, chunking utilities and tests
