#!/bin/bash
set -euo pipefail

base="${1:-origin/main}"
head="${2:-HEAD}"

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

# Build the test image
docker build -t datacreek:test .

mapfile -t test_files < <(git diff --name-only "$base" "$head" -- 'tests/*.py')
mapfile -t cov_files < <(git diff --name-only "$base" "$head" -- 'datacreek/**/*.py')
cmd=(pytest --cov-report=xml --cov-fail-under=80 -q)
if [ "${#cov_files[@]}" -eq 0 ]; then
  cmd+=(--cov=datacreek)
else
  for f in "${cov_files[@]}"; do
    mod=${f%.py}
    mod=${mod//\//.}
    cmd+=(--cov="$mod")
  done
fi
log_file="pytest.log"
if [ "${#test_files[@]}" -gt 0 ]; then
  echo "Running changed tests: ${test_files[*]}"
  cmd+=("${test_files[@]}")
else
  echo "No changed tests found, running full test suite"
fi

set +e
docker run --rm \
  -v "$DIR":/workspace \
  -w /workspace datacreek:test \
  bash -c "pip install pytest pytest-cov && PYTHONPATH=/workspace ${cmd[*]} -vv 2>&1 | tee /workspace/$log_file"
status=$?
set -e
if [ $status -ne 0 ]; then
  echo 'Tests failed. Output:'
  cat "$log_file"
fi
exit $status
