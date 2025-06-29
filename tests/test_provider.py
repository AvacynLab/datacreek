import requests
from synthetic_data_kit.utils import load_config
from synthetic_data_kit.utils.provider import resolve_provider, check_vllm_server


def test_resolve_provider_vllm():
    cfg = load_config("configs/config.yaml")
    provider, api_base, model = resolve_provider(cfg, provider="vllm")
    assert provider == "vllm"
    assert isinstance(api_base, str)
    assert isinstance(model, str)


def test_check_vllm_server_false(monkeypatch):
    def fake_get(url, timeout):
        raise requests.exceptions.RequestException

    monkeypatch.setattr(requests, "get", fake_get)
    assert check_vllm_server("http://localhost:1234") is False
