#!/bin/bash
set -euo pipefail

base="${1:-origin/main}"
head="${2:-HEAD}"

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

# Build the test image
docker build -t datacreek:test .

mapfile -t test_files < <(git diff --name-only "$base" "$head" -- 'tests/*.py')
cmd=(pytest -q)
if [ "${#test_files[@]}" -gt 0 ]; then
  echo "Running changed tests: ${test_files[*]}"
  cmd+=("${test_files[@]}")
else
  echo "No changed tests found, running full test suite"
fi

docker run --rm \
  -v "$DIR":/workspace \
  -w /workspace datacreek:test \
  bash -c "pip install pytest && ${cmd[*]}"
