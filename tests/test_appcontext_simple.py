import importlib
from pathlib import Path
import types
import sys

sys.path.insert(0, ".")


def test_default_path(monkeypatch):
    dummy_cfg = types.SimpleNamespace(DEFAULT_CONFIG_PATH=Path('dflt.yaml'))
    monkeypatch.setitem(importlib.sys.modules, 'datacreek.utils.config', dummy_cfg)
    app = importlib.reload(importlib.import_module('datacreek.core.context'))
    ctx = app.AppContext()
    assert ctx.config_path == Path('dflt.yaml')
    assert ctx.config == {}


def test_custom_path(tmp_path):
    from datacreek.core.context import AppContext
    custom = tmp_path / 'cfg.yml'
    ctx = AppContext(custom)
    assert ctx.config_path == custom
    assert ctx.config == {}
