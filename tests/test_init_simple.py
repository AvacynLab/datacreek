import importlib

import datacreek


def test_version_exposed():
    assert hasattr(datacreek, "__version__")


def test_dynamic_getattr(monkeypatch):
    def fake_md5(data):
        return "ok"

    monkeypatch.setattr("datacreek.utils.checksum.md5_file", fake_md5)
    assert datacreek.md5_file(b"x") == "ok"
