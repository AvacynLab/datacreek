import os
from importlib import reload
from pathlib import Path
import pytest

import datacreek.utils.config as cfg


def _write_config(tmp_path: Path) -> str:
    content = """
llm:
  provider: api-endpoint
vllm:
  model: llama
api-endpoint:
  api_key: 123
  max_retries: 2
  retry_delay: 0.5
curate:
  threshold: 0.8
  batch_size: 16
format:
  default: jsonl
  include_metadata: true
prompts:
  greet: Hello
    """
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    return str(path)


def test_load_and_helpers(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setattr(cfg, "yaml", None, raising=False)
    data = cfg.load_config(path)
    assert data["llm"]["provider"] == "api-endpoint"
    assert data["curate"]["batch_size"] == 16
    assert cfg.get_llm_provider(data) == "api-endpoint"
    monkeypatch.setenv("LLM_MODEL", "foo")
    vllm = cfg.get_vllm_settings(data)
    assert vllm.model == "foo"
    openai = cfg.get_openai_settings(data)
    assert openai.api_key == 123
    gen = cfg.get_generation_config(data)
    assert gen.retrieval_top_k == 3  # default from dataclass
    fmt = cfg.get_format_settings(data)
    assert fmt.default == "jsonl"
    merged = cfg.merge_configs({"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2}
    over = cfg.load_config_with_overrides(path, {"new": 3})
    assert over["new"] == 3


def test_watcher_cycle(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setattr(cfg, "yaml", None, raising=False)
    cfg.stop_config_watcher()
    cfg.start_config_watcher(path)
    assert cfg._config_observer is not None
    cfg.stop_config_watcher()
    assert cfg._config_observer is None


def test_load_config_fallback(monkeypatch, tmp_path):
    path = _write_config(tmp_path)
    monkeypatch.setenv(cfg.CONFIG_PATH_ENV, path)
    monkeypatch.setattr(cfg, "yaml", None, raising=False)
    data = cfg.load_config()
    assert data["api-endpoint"]["api_key"] == 123


def test_load_config_missing(monkeypatch):
    monkeypatch.delenv(cfg.CONFIG_PATH_ENV, raising=False)
    monkeypatch.setattr(cfg, "yaml", None, raising=False)
    calls = []

    def fake_exists(p):
        calls.append(p)
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError):
        cfg.load_config()
    assert calls
