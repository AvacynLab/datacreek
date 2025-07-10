# Agent Tasks Checklist

## Pending Tasks

- [x] Integrate Node2Vec via Neo4j GDS
- [x] Build FAISS index from Node2Vec vectors
- [x] Extend AutoTuneState with p, q, d
- [x] Update autotune_step to handle Node2Vec params
- [x] Document Node2Vec + FAISS
- [x] Add DatasetBuilder wrappers for Node2Vec GDS and FAISS index

## Info
This project is a SaaS pipeline for dataset generation. It ingests multimodal documents, constructs a fractal hypergraph, enforces topological coherence, and uses an autotuning loop to balance entropy, bottleneck distance, IB loss and MDL. Generation is guided by LLMs and data is exported with traceability.

## History
- commit 4a046b2 replaced task list with No tasks
- commit 46f1749 added autotune step, crypto helpers, chunking utilities and tests
- added Node2Vec GDS method, FAISS index, autotune params and docs
- added dataset wrappers and tests for Node2Vec GDS and FAISS index
- installed missing dependencies and executed Node2Vec tests
- installed fakeredis, networkx, numpy and requests for tests
- installed pyyaml, pydantic, rich, neo4j, python-dateutil, SQLAlchemy and fastapi
  to satisfy imports; ran selective tests (1 passed, 1 skipped)
