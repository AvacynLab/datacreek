import os
import pytest

from datacreek.utils.config import load_config, get_generation_config
from datacreek.config_models import GenerationSettingsModel, GenerationSettings


def test_env_override(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("GEN_TEMPERATURE", "0.5")
    gen_cfg = get_generation_config(cfg)
    assert gen_cfg.temperature == 0.5


def test_generation_settings_model_validation():
    model = GenerationSettingsModel(temperature=0.4)
    settings = model.to_settings()
    assert isinstance(settings, GenerationSettings)
    assert settings.temperature == 0.4
    with pytest.raises(Exception):
        GenerationSettingsModel(temperature="hot")
