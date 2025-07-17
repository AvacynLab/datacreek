"""Utility helpers for datacreek."""

from .chunking import chunk_by_sentences, chunk_by_tokens
from .config import (
    get_curate_config,
    get_curate_settings,
    get_format_config,
    get_format_settings,
    get_generation_config,
    get_llm_settings,
    get_openai_config,
    get_openai_settings,
    get_prompt,
    get_vllm_config,
    get_vllm_settings,
    load_config,
    merge_configs,
)
from .crypto import decrypt_pii_fields, encrypt_pii_fields, xor_decrypt, xor_encrypt
from .dataset_cleanup import deduplicate_pairs
from .gitinfo import get_commit_hash
from .graph_text import graph_to_text, neighborhood_to_sentence, subgraph_to_text
from .llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
    qa_pairs_to_records,
)
from .metrics import push_metrics
from .progress import create_progress, progress_context
from .cache import cache_l1
from .redis_helpers import decode_hash
from .text import clean_text, extract_json_from_text, normalize_units, split_into_chunks
from .toolformer import execute_tool_calls, insert_tool_calls


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
    "get_prompt",
    "merge_configs",
    "split_into_chunks",
    "chunk_by_tokens",
    "chunk_by_sentences",
    "extract_json_from_text",
    "clean_text",
    "normalize_units",
    "parse_qa_pairs",
    "parse_ratings",
    "create_progress",
    "progress_context",
    "convert_to_conversation_format",
    "qa_pairs_to_records",
    "deduplicate_pairs",
    "get_commit_hash",
    "extract_facts",
    "extract_entities",
    "decode_hash",
    "neighborhood_to_sentence",
    "subgraph_to_text",
    "graph_to_text",
    "insert_tool_calls",
    "execute_tool_calls",
    "xor_encrypt",
    "xor_decrypt",
    "encrypt_pii_fields",
    "decrypt_pii_fields",
    "push_metrics",
    "cache_l1",
]
