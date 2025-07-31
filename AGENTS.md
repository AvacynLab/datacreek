# Development Log

## Tasks
- [x] Heavy coverage >80% for `utils.image_dedup`, `utils.kafka_queue`, `utils.redis_pid`
- [x] Heavy coverage >80% for `utils.retrieval`, `utils.text`
- [x] Heavy coverage >80% for `utils.metrics`, `utils.progress`
- [x] Maintain overall unit test coverage between 70% and 80%
- [x] Improve coverage for other utilities (e.g. modality)
- [x] Document coverage command and results
- [x] Heavy coverage >80% for `utils.batch`

## Coverage
Run tests with:
```bash
bash run_cov.sh
```

Latest results (from `bash run_cov.sh`):
```
Unit tests: TOTAL 7289 statements, 4225 missed, 42% coverage
FAILED tests/unit/test_batch.py::test_async_process_batches_success
FAILED tests/unit/test_batch.py::test_async_process_batches_error
FAILED tests/unit/test_batch.py::test_async_process_batches_parse_error
FAILED tests/unit/test_batch.py::test_async_process_batches_raise_on_error
FAILED tests/unit/test_config_utils.py::test_format_and_model_profiles
FAILED tests/unit/test_conversation_generator.py::test_conversation_generator_async
... (truncated)

Heavy tests: TOTAL 7289 statements, 5466 missed, 25% coverage
FAILED tests/heavy/test_config_utils_heavy.py::test_format_and_model_profiles
FAILED tests/heavy/test_dataset_full.py::test_generation_layer_async
FAILED tests/heavy/test_dataset_full.py::test_ingest_file_async
FAILED tests/heavy/test_misc_analysis.py::test_sheaf_sla
FAILED tests/heavy/test_redis_pid.py::test_update_ttl
FAILED tests/heavy/test_redis_pid.py::test_pid_loop
ERROR tests/heavy/test_ann_cpu.py
ERROR tests/heavy/test_autotune_heavy.py
... (truncated)
```
