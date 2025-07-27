# Task List
- [ ] Ensure overall coverage >= 80%
- [x] Add unit tests to improve coverage file by file
- [x] Maintain organized test directories for unit, integration, e2e, benchmark, property, heavy and golden tests
- [x] Increase cache.py coverage to >=80%
- [x] Increase backpressure.py coverage to >=80%
- [x] Increase curate.py coverage to >=80%
- [x] Increase batch.py coverage to >=80%
- [x] Increase image_dedup.py coverage to >=80%
- [x] Increase save_as.py coverage to >=80%
- [x] Increase core/fractal.py coverage to >=80%
- [x] Increase models/llm_client.py coverage to >=80%
- [x] Increase services.py coverage to >=80%
- [x] Increase utils/crypto.py coverage to >=80%
- [x] Increase security/dp_middleware.py coverage to >=80%
- [x] Increase core/dataset_light.py coverage to >=80%
- [x] Increase utils/audio_vad.py coverage to >=80%
- [x] Increase utils/checksum.py coverage to >=80%
- [x] Increase utils/curation_agent.py coverage to >=80%
- [x] Increase backend/array_api.py coverage to >=80%

# History
- Installed missing dependencies and configured PYTHONPATH for coverage
- Added unit tests for models.llm_client covering vLLM server checks and request logic
- Coverage baseline: 51% overall
- Coverage after llm_client tests: 52% overall with models/llm_client.py at 81%
- Added explain_router unit tests covering auth, dataset loading and public endpoint
- Coverage after explain_router tests: 52% overall with routers/explain_router.py at 100%
- Added services unit tests using FakeRedis and in-memory SQLite
- Coverage after services tests: 53% overall with services.py at 88%
- Added runners unit tests covering node2vec, graphwave and poincare
- Coverage after runners tests: 54% overall with core/runners.py at 86%
- Baseline before crypto tests: 54% overall
- Added crypto unit tests for XOR helpers and PII field utilities
- Coverage after crypto tests: 54% overall with utils/crypto.py at 100%
- Added llm_service unit tests to wrap LLMClient and verify sync and async helpers
- Coverage after llm_service tests: 54% overall with models/llm_service.py at 100%
- Added dp_middleware unit tests using Starlette TestClient and dummy DB
- Coverage command: python -m pytest -q tests/unit/test_autotune.py tests/unit/test_docx_parser.py tests/unit/test_html_parser.py tests/unit/test_image_parser.py tests/unit/test_pdf_parser.py tests/unit/test_ppt_parser.py tests/unit/test_txt_parser.py tests/unit/test_whisper_youtube_parsers.py tests/unit/test_dp_middleware.py --cov=datacreek
- Coverage after dp_middleware tests: 14% overall with security/dp_middleware.py at 100%
- Installed required dependencies (networkx, numpy, pydantic, etc.) so the full
  unit suite runs without import errors
- Fixed dp_middleware tests by ensuring the datacreek.db module is properly
  patched during execution
- Coverage command: PYTHONPATH=$(pwd) pytest -q tests/unit --cov=datacreek
- Coverage after running all unit tests: 54% overall
- Added dataset_light, audio_vad, checksum, curation_agent and array_api tests
- Installed pytest-cov, pydantic, networkx and numpy
- Coverage command: python -m pytest -q tests/unit/test_autotune.py tests/unit/test_docx_parser.py tests/unit/test_html_parser.py tests/unit/test_image_parser.py tests/unit/test_pdf_parser.py tests/unit/test_ppt_parser.py tests/unit/test_txt_parser.py tests/unit/test_whisper_youtube_parsers.py tests/unit/test_dataset_light.py tests/unit/test_audio_vad.py tests/unit/test_checksum.py tests/unit/test_curation_agent.py tests/unit/test_array_api.py --cov=datacreek
- Coverage after new tests: dataset_light.py 95%, audio_vad.py 100%, checksum.py 100%, curation_agent.py 83%, array_api.py 86%, total 15%
