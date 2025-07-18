# Changelog

## [1.1] - Unreleased
### Added
- Prometheus alert rules for Redis hit ratio and eigsh timeouts
- Model card now embeds the code commit SHA
- LMDB eviction logging with ring buffer and cause tracking
- Parallel BLIP captioning using thread pool
- Async PID controller adjusting Redis TTL
- Async TTL manager with error handling
- Tenant privacy budgets via SQL
- Ingestion backpressure with bounded queue
- GPU batch transcription for Whisper.cpp
- Normalized HAA index with migration script
- Rollback scripts for datastore snapshots
- nprobe Bayesian autotuning
- Node2Vec hyperparameter autotuner
- Poincaré embedding recentering
- Dynamic GraphWave bandwidth estimation
- Comprehensive benchmark script `bench_all.py`
- GPU-accelerated GraphWave using cuSPARSE kernels
- LLM curation agent powered by Langchain

### Changed
- Multiple ingestion and caching utilities for robustness

