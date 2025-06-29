"""Utility helpers for datacreek."""

from .config import (
    get_curate_config,
    get_curate_settings,
    get_format_config,
    get_format_settings,
    get_generation_config,
    get_llm_settings,
    get_openai_config,
    get_openai_settings,
    get_output_paths,
    get_path_config,
    get_prompt,
    get_vllm_config,
    get_vllm_settings,
    load_config,
    merge_configs,
)
from .dataset_cleanup import deduplicate_pairs
from .llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
    qa_pairs_to_records,
)
from .progress import create_progress, progress_context
from .text import clean_text, extract_json_from_text, split_into_chunks


def __getattr__(name: str):
    """Lazily import heavy utilities to avoid circular imports."""
    if name == "extract_facts":
        from .fact_extraction import extract_facts as func

        return func
    if name == "extract_entities":
        from .entity_extraction import extract_entities as func

        return func
    raise AttributeError(name)


__all__ = [
    "load_config",
    "get_path_config",
    "get_vllm_config",
    "get_vllm_settings",
    "get_openai_config",
    "get_openai_settings",
    "get_llm_settings",
    "get_generation_config",
    "get_curate_config",
    "get_curate_settings",
    "get_format_config",
    "get_format_settings",
    "get_output_paths",
    "get_prompt",
    "merge_configs",
    "split_into_chunks",
    "extract_json_from_text",
    "clean_text",
    "parse_qa_pairs",
    "parse_ratings",
    "create_progress",
    "progress_context",
    "convert_to_conversation_format",
    "qa_pairs_to_records",
    "deduplicate_pairs",
    "extract_facts",
    "extract_entities",
]
