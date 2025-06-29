import pathlib

from synthetic_data_kit.utils.config_models import AppConfig, load_config_model


def test_load_config_model():
    cfg = load_config_model(pathlib.Path("configs/config.yaml"))
    assert isinstance(cfg, AppConfig)
    # basic sanity checks
    assert cfg.llm.provider in {"vllm", "api-endpoint"}
    assert cfg.generation.num_pairs > 0
