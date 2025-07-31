# Development Log

## Tasks
- [x] Heavy coverage >80% for `utils.image_dedup`, `utils.kafka_queue`, `utils.redis_pid`
- [x] Heavy coverage >80% for `utils.retrieval`, `utils.text`
- [x] Heavy coverage >80% for `utils.metrics`, `utils.progress`
- [x] Maintain overall unit test coverage between 70% and 80%
- [x] Improve coverage for other utilities (e.g. modality)
- [x] Document coverage command and results
- [x] Heavy coverage >80% for `utils.batch`
- [x] Ensure environment has dependencies for running heavy tests
- [x] Heavy coverage >80% for `utils.config` and `config_models`

## Coverage
Run tests with:
```bash
bash run_cov.sh
```

Latest results (`bash run_cov.sh`):
```text
TOTAL 15257 statements, 2463 missed, 84% coverage
```

## History
- Installed requests and added heavy tests for `LLMClient`
- Heavy coverage improved to 84%
