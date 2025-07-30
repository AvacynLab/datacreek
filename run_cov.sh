#!/bin/bash
# Run full unit and heavy test suites for coverage.
# Always execute every test under tests/unit and tests/heavy.
set -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Run unit tests
rm -f .coverage coverage.log
PYTHONPATH="$(pwd)" pytest -q tests/unit --continue-on-collection-errors \
  --cov=datacreek --cov-report=term > coverage.log || true

# Run heavy tests separately
rm -f .coverage
PYTHONPATH="$(pwd)" pytest -q tests/heavy --continue-on-collection-errors \
  --cov=datacreek --cov-report=term >> coverage.log || true
