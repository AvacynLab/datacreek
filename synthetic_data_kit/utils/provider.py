from typing import Optional, Tuple

import requests

from .config import (
    get_llm_provider,
    get_openai_config,
    get_vllm_config,
)


def resolve_provider(
    config,
    provider: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Resolve provider and return (provider, api_base, model)."""
    selected = provider or get_llm_provider(config)
    if selected == "api-endpoint":
        endpoint_cfg = get_openai_config(config)
        api_base = api_base or endpoint_cfg.get("api_base")
        model = model or endpoint_cfg.get("model")
    else:
        vllm_cfg = get_vllm_config(config)
        api_base = api_base or vllm_cfg.get("api_base")
        model = model or vllm_cfg.get("model")
    return selected, api_base, model


def check_vllm_server(api_base: str) -> bool:
    """Return True if vLLM server responds to /models."""
    try:
        resp = requests.get(f"{api_base}/models", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
