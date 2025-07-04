#!/usr/bin/env bash
set -euo pipefail
BASE=${1:-origin/main}
HEAD=${2:-HEAD}

changed=$(git diff --name-only "$BASE" "$HEAD")
files=()
while IFS= read -r file; do
  if [[ "$file" == tests/*.py ]]; then
    files+=("$file")
  fi
done <<< "$changed"
if [ ${#files[@]} -eq 0 ]; then
  echo "No changed test files, running full suite" >&2
  exec pytest -q
else
  echo "Running changed tests: ${files[*]}" >&2
  exec pytest -q "${files[@]}"
fi
