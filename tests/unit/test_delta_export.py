import json
from pathlib import Path

import datacreek.utils.delta_export as de


def test_export_delta_list(tmp_path):
    data = [{"a": 1}, {"b": 2}]
    path = de.export_delta(data, root=str(tmp_path), org_id=1, kind="foo")
    assert path.name == "data.jsonl"
    content = path.read_text().splitlines()
    assert [json.loads(c) for c in content] == data


def test_export_delta_string(tmp_path):
    path = de.export_delta("text", root=str(tmp_path), org_id="x", kind="bar")
    assert path.read_text() == "text"


def test_lakefs_commit(monkeypatch):
    called = {}
    def fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check
    monkeypatch.setattr(de.subprocess, "run", fake_run)
    de.lakefs_commit(Path("/tmp/file"), repo="r")
    assert called["cmd"] == ["lakefs", "commit", "r", "-m", "export /tmp/file"]
    assert called["check"] is True


def test_lakefs_commit_error(monkeypatch):
    def bad_run(*a, **k):
        raise RuntimeError
    monkeypatch.setattr(de.subprocess, "run", bad_run)
    de.lakefs_commit(Path("/tmp/file"), repo="r")
