import sys
sys.path.insert(0, "./")
import builtins
import subprocess
from datacreek.utils import gitinfo


def test_get_commit_hash_success(monkeypatch):
    """Ensure commit hash is returned when git command succeeds."""
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, cwd=None: b"abcd1234\n")
    assert gitinfo.get_commit_hash() == "abcd1234"


def test_get_commit_hash_failure(monkeypatch):
    """Ensure 'unknown' is returned when git command fails."""
    def raise_error(cmd, cwd=None):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", raise_error)
    assert gitinfo.get_commit_hash() == "unknown"
