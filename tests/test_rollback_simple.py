import subprocess
from pathlib import Path

import pytest

from datacreek.analysis.rollback import rollback_gremlin_diff, SheafSLA


def test_rollback_diff(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    # intercept git diff call
    called = {}
    def fake_check_output(cmd, cwd=None):
        called['cmd'] = cmd
        called['cwd'] = cwd
        return b"diff-data"
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    out = rollback_gremlin_diff(str(repo))
    assert Path(out).read_text() == "diff-data"
    assert called['cmd'] == ["git", "diff", "HEAD~1", "HEAD"]
    assert called['cwd'] == repo


def test_sheaf_sla():
    sla = SheafSLA(threshold_hours=1)
    sla.record_failure(1)
    sla.record_failure(3601)
    sla.record_failure(7201)
    # two intervals => mean = 3600
    assert pytest.approx(sla.mttr_hours()) == 1.0
    assert sla.sla_met()
    sla.threshold = 0.5 * 3600
    assert not sla.sla_met()


def test_mttr_insufficient_failures():
    sla = SheafSLA()
    sla.record_failure(42)
    assert sla.mttr_hours() == 0.0
