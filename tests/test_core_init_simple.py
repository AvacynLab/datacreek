import importlib
import sys
import types
sys.path.insert(0, ".")
import pytest


def test_lazy_load_appcontext(monkeypatch):
    dummy_cfg = types.SimpleNamespace(DEFAULT_CONFIG_PATH="cfg.yml")
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", dummy_cfg)
    sys.modules.pop("datacreek.core.context", None)
    core = importlib.reload(importlib.import_module("datacreek.core"))
    assert "datacreek.core.context" not in sys.modules
    AppContext = core.AppContext
    assert "datacreek.core.context" in sys.modules
    from datacreek.core.context import AppContext as Expected
    assert AppContext is Expected


def test_getattr_invalid():
    core = importlib.reload(importlib.import_module("datacreek.core"))
    with pytest.raises(AttributeError):
        core.__getattr__("missing")


def test_all_contains_appcontext():
    import datacreek.core as core
    assert "AppContext" in core.__all__
