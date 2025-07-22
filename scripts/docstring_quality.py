#!/usr/bin/env python3
"""Simple docstring quality checker using interrogate."""
import re
import subprocess
import sys

TARGET = 0.80


def main() -> int:
    cmd = [
        "interrogate",
        "-e",
        "datacreek/api.py",
        "-e",
        "datacreek/db.py",
        "-e",
        "datacreek/pipelines.py",
        "-e",
        "datacreek/schemas.py",
        "-e",
        "datacreek/services.py",
        "-e",
        "datacreek/storage.py",
        "-e",
        "datacreek/tasks.py",
        "-e",
        "datacreek/config_models.py",
        "-e",
        "datacreek/utils",
        "datacreek",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout.strip().splitlines()
    coverage = None
    for line in output:
        match = re.search(r"actual: ([0-9.]+)%", line)
        if match:
            coverage = float(match.group(1)) / 100.0
            break
    if coverage is None:
        print("Failed to parse coverage")
        print(proc.stdout)
        print(proc.stderr)
        return 1
    print(f"docstring quality: {coverage:.2f}")
    return 0 if coverage >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
