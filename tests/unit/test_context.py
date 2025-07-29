from datacreek.core.context import AppContext
from datacreek.utils.config import DEFAULT_CONFIG_PATH


def test_appcontext_defaults():
    ctx = AppContext()
    assert ctx.config_path == DEFAULT_CONFIG_PATH
    assert ctx.config == {}


def test_appcontext_custom_path(tmp_path):
    custom = tmp_path / "cfg.yaml"
    ctx = AppContext(custom)
    assert ctx.config_path == custom
