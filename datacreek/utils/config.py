# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Config Utilities
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from datacreek.config_models import (
    CurateSettings,
    FormatSettings,
    GenerationSettings,
    LLMSettings,
    OpenAISettings,
    OutputPaths,
    VLLMSettings,
)

# Default config location relative to the package (original)
ORIGINAL_CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml"
    )
)

# Add fallback location inside the package (recommended for installed packages)
PACKAGE_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
)

# Use internal package path as default
DEFAULT_CONFIG_PATH = PACKAGE_CONFIG_PATH

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if config_path is None:
        # Try each path in order until one exists
        for path in [PACKAGE_CONFIG_PATH, ORIGINAL_CONFIG_PATH]:
            if os.path.exists(path):
                config_path = path
                break
        else:
            # If none exists, use the default (which will likely fail, but with a clear error)
            config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        # Support relative paths when tests run from temporary directories
        if not os.path.isabs(config_path):
            pkg_root = Path(__file__).resolve().parents[2]
            alt_path = pkg_root / config_path
            if os.path.exists(alt_path):
                config_path = str(alt_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    logger.info("Loading config from: %s", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Debug: Print LLM provider if it exists
    if "llm" in config and "provider" in config["llm"]:
        logger.info("Config has LLM provider set to: %s", config["llm"]["provider"])
    else:
        logger.info("Config does not have LLM provider set")

    return config


def get_path_config(config: Dict[str, Any], path_type: str, file_type: Optional[str] = None) -> str:
    """Get path from configuration based on type and optionally file type"""
    paths = config.get("paths", {})

    if path_type == "input":
        input_paths = paths.get("input", {})
        if file_type and file_type in input_paths:
            return input_paths[file_type]
        return input_paths.get("default", "data/input")

    elif path_type == "output":
        output_paths = paths.get("output", {})
        if file_type and file_type in output_paths:
            return output_paths[file_type]
        return output_paths.get("default", "data/output")

    else:
        raise ValueError(f"Unknown path type: {path_type}")


def get_llm_provider(config: Dict[str, Any]) -> str:
    """Get the selected LLM provider

    Returns:
        String with provider name: 'vllm' or 'api-endpoint'
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "vllm")
    logger.debug("get_llm_provider returning: %s", provider)
    if (
        provider != "api-endpoint"
        and "llm" in config
        and "provider" in config["llm"]
        and config["llm"]["provider"] == "api-endpoint"
    ):
        logger.warning("Config has 'api-endpoint' but returning '%s'", provider)
    return provider


def get_llm_settings(config: Dict[str, Any]) -> LLMSettings:
    """Return general LLM configuration as :class:`LLMSettings`."""

    llm_cfg = config.get("llm", {})
    defaults = {"provider": "vllm"}
    defaults.update(llm_cfg)
    return LLMSettings.from_dict(defaults)


def get_vllm_settings(config: Dict[str, Any]) -> VLLMSettings:
    """Return VLLM configuration as :class:`VLLMSettings`."""

    defaults = config.get(
        "vllm",
        {
            "api_base": "http://localhost:8000/v1",
            "port": 8000,
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    ).copy()
    for field_name in VLLMSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(VLLMSettings(), field_name))
    return VLLMSettings.from_dict(defaults)


def get_vllm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backwards compatible wrapper returning a plain dictionary."""

    return get_vllm_settings(config).__dict__


def get_openai_settings(config: Dict[str, Any]) -> OpenAISettings:
    """Return OpenAI/API endpoint configuration as :class:`OpenAISettings`."""

    defaults = config.get(
        "api-endpoint",
        {
            "api_base": None,
            "api_key": None,
            "model": "gpt-4o",
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    ).copy()
    for field_name in OpenAISettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(OpenAISettings(), field_name))
    return OpenAISettings.from_dict(defaults)


def get_openai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backwards compatible wrapper returning a plain dictionary."""

    return get_openai_settings(config).__dict__


def _env_override(key: str) -> Optional[str]:
    """Helper to fetch environment variable overrides."""
    env_key = f"GEN_{key.upper()}"
    return os.environ.get(env_key)


def get_generation_config(config: Dict[str, Any]) -> GenerationSettings:
    """Return generation configuration as :class:`GenerationSettings`."""

    defaults = config.get("generation", {}).copy()

    # Fill in defaults from dataclass definition
    for field_name, field_def in GenerationSettings.__dataclass_fields__.items():
        defaults.setdefault(field_name, getattr(GenerationSettings(), field_name))

    # Apply environment variable overrides if present for all known keys
    env_overrides = {
        k: type(defaults.get(k))(v)  # type: ignore
        for k in GenerationSettings.__dataclass_fields__.keys()
        if (v := _env_override(k)) is not None
    }

    cfg = {**defaults, **env_overrides}
    return GenerationSettings.from_dict(cfg)


def get_curate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get curation configuration"""
    return config.get("curate", {"threshold": 7.0, "batch_size": 8, "temperature": 0.1})


def get_curate_settings(config: Dict[str, Any]) -> CurateSettings:
    """Return curation configuration as :class:`CurateSettings`."""
    defaults = config.get("curate", {}).copy()
    for field_name in CurateSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(CurateSettings(), field_name))
    return CurateSettings.from_dict(defaults)


def get_format_settings(config: Dict[str, Any]) -> FormatSettings:
    """Return output formatting configuration as :class:`FormatSettings`."""
    defaults = config.get("format", {}).copy()
    for field_name in FormatSettings.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(FormatSettings(), field_name))
    return FormatSettings.from_dict(defaults)


def get_output_paths(config: Dict[str, Any]) -> OutputPaths:
    """Return output path settings as :class:`OutputPaths`."""
    defaults = config.get("paths", {}).get("output", {}).copy()
    for field_name in OutputPaths.__dataclass_fields__:
        defaults.setdefault(field_name, getattr(OutputPaths(), field_name))
    return OutputPaths.from_dict(defaults)


def get_format_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get format configuration"""
    return config.get("format", {"default": "jsonl", "include_metadata": True, "pretty_json": True})


def get_prompt(config: Dict[str, Any], prompt_name: str) -> str:
    """Get prompt by name"""
    prompts = config.get("prompts", {})
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
    return prompts[prompt_name]


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries"""
    result = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_overrides(
    config_path: str | None = None, overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Convenience wrapper around :func:`load_config` applying ``overrides``."""

    cfg = load_config(config_path)
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return cfg


def get_model_profile(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Retrieve a model profile by name."""
    profiles = config.get("models", {})
    if name not in profiles:
        raise KeyError(f"Model profile '{name}' not found")
    return profiles[name]


def get_redis_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("databases", {}).get("redis", {"host": "localhost", "port": 6379})


def get_neo4j_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("databases", {}).get(
        "neo4j",
        {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "neo4j"},
    )
