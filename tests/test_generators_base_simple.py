import os
from pathlib import Path
import yaml

import importlib
import sys
import datacreek.generators.base as base

def reload_config_module():
    """Reload the real config module in case a previous test patched it."""
    if 'datacreek.utils.config' in sys.modules:
        cfg = importlib.reload(sys.modules['datacreek.utils.config'])
    else:
        cfg = importlib.import_module('datacreek.utils.config')
    sys.modules['datacreek.utils.config'] = cfg
    return cfg

class DummyClient:
    def __init__(self, config):
        self.config = config


def write_config(tmp_path, data):
    path = tmp_path / 'config.yaml'
    with open(path, 'w') as f:
        yaml.dump(data, f)
    return path


def test_base_generator_defaults(tmp_path):
    reload_config_module()
    client = DummyClient({'generation': {'temperature': 0.2}})
    gen = base.BaseGenerator(client)
    assert gen.config == client.config
    assert gen.generation_config.temperature == 0.2


def test_base_generator_overrides_and_path(tmp_path):
    reload_config_module()
    cfg_path = write_config(tmp_path, {'generation': {'temperature': 0.5}})
    overrides = {'generation': {'temperature': 0.8}}
    client = DummyClient({'generation': {'temperature': 0.3}})
    gen = base.BaseGenerator(client, config_path=cfg_path, config_overrides=overrides)
    assert gen.config['generation']['temperature'] == 0.8
    assert gen.generation_config.temperature == 0.8


def test_base_generator_env_override(tmp_path, monkeypatch):
    reload_config_module()
    cfg_path = write_config(tmp_path, {'generation': {'batch_size': 4}})
    monkeypatch.setenv('DATACREEK_CONFIG', str(cfg_path))
    overrides = {'generation': {'batch_size': 10}}
    gen = base.BaseGenerator(DummyClient({}), config_overrides=overrides)
    assert gen.config['generation']['batch_size'] == 10
    assert gen.generation_config.batch_size == 10
