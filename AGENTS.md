# Task List

- [x] Investigate low coverage in CI
- [x] Install required dependencies for full test suite
- [x] Add tests for utility modules
- [ ] Ensure overall coverage >= 80%
- [x] Add tests for backpressure utilities
- [x] Add tests for checksum utilities
- [x] Add tests for metrics helper
- [x] Add tests for rate limit local fallback

Current objective: expand unit tests for lightweight modules to incrementally raise overall repository coverage toward 80%. Each file is tackled separately with simple tests avoiding heavy dependencies.

History:
- Added initial tests for dataset cleanup, delta export, gitinfo, chunking, audio VAD, crypto, and redis helpers
- Wrote additional tests for backpressure, checksum, metrics, and rate limit modules
