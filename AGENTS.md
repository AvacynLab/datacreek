# Task List

- [x] Investigate low coverage in CI
- [ ] Install required dependencies for full test suite (pytest-cov installed)
- [x] Add tests for utility modules
- [ ] Ensure overall coverage >= 80%

Current objective: achieve 80% coverage across the repository. Start by adding lightweight tests for modules that do not require heavy dependencies and run them under pytest with coverage.
\nHistory: added tests for chunking, crypto, and redis helpers; coverage for utils passes 87%.

Latest coverage (tests/test_simple_utils.py tests/test_audio_vad.py tests/test_more_utils.py):

```
TOTAL                                            15422  14765     4%
============================== 17 passed in 1.77s ==============================
```

