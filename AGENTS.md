# Development Log

## Tasks
- [x] Heavy coverage >80% for `utils.image_dedup`, `utils.kafka_queue`, `utils.redis_pid`
- [x] Heavy coverage >80% for `utils.retrieval`, `utils.text`
- [x] Heavy coverage >80% for `utils.metrics`, `utils.progress`
- [x] Maintain overall unit test coverage between 70% and 80%
- [ ] Improve coverage for other utilities (e.g. modality)
- [x] Document coverage command and results

## Coverage
Run tests with:
```bash
bash run_cov.sh
```

Latest results:
```
Unit tests: TOTAL 7289 statements, 1919 missed, 74% coverage
errors due to missing deps but coverage collected

Heavy tests: TOTAL 7289 statements, 3559 missed, 51% coverage
139 passed, 3 skipped
```
