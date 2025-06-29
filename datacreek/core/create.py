# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.generators.qa_generator import QAGenerator
from datacreek.generators.vqa_generator import VQAGenerator
from datacreek.models.content_type import ContentType
from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_generation_config, get_output_paths, load_config

logger = logging.getLogger(__name__)


def load_document_text(file_path: str) -> str:
    """Return the raw text from ``file_path``."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_output_dir(config_path: Optional[Path]) -> str:
    """Return the default generated output directory from config."""
    cfg = load_config(config_path)
    paths = get_output_paths(cfg)
    return paths.generated


def init_llm_client(
    config_path: Optional[Path],
    *,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMClient:
    """Instantiate :class:`LLMClient` with common parameters."""
    return LLMClient(
        config_path=config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model_name=model_name,
    )


def _base_name(file_path: Optional[str]) -> str:
    return os.path.splitext(os.path.basename(file_path))[0] if file_path else "input"


def process_file(
    file_path: Optional[str],
    output_dir: Optional[str],
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: ContentType | str = ContentType.QA,
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    *,
    async_mode: bool = False,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    document_text: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Process a file to generate content

    Args:
        file_path: Path to the text file to process
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate. Can be a
            :class:`~datacreek.models.content_type.ContentType` value or string
            ("qa", "summary", "cot", "cot-enhance", "vqa_add_reasoning").
        num_pairs: Target number of QA pairs to generate
        async_mode: Use asynchronous LLM requests when supported

    Returns:
        Path to the output file
    """
    save_to_file = output_dir is not None
    if output_dir is None:
        output_dir = resolve_output_dir(config_path)
    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)

    client = init_llm_client(
        config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model_name=model,
    )

    logger.info("Using %s provider", client.provider)

    # Generate base filename for output
    base_name = _base_name(file_path)

    ct = ContentType(content_type)

    def _generate_qa() -> Any:
        generator = QAGenerator(client, config_path, config_overrides=config_overrides)

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs_local = generation_config.num_pairs
        else:
            num_pairs_local = num_pairs

        if async_mode:
            gen_result = asyncio.run(
                generator.process_document_async(
                    document_text,
                    num_pairs=num_pairs_local,
                    verbose=verbose,
                )
            )
        else:
            gen_result = generator.process_document(
                document_text,
                num_pairs=num_pairs_local,
                verbose=verbose,
            )

        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
            logger.info("Saving result to %s", output_path)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(gen_result.to_dict(), f, indent=2)
                logger.info("Successfully wrote result to %s", output_path)
            except Exception as e:
                logger.error("Error writing result file: %s", e)
            return output_path
        return gen_result.to_dict()

    def _generate_summary() -> Any:
        generator = QAGenerator(client, config_path, config_overrides=config_overrides)

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        summary = generator.generate_summary(document_text, verbose=verbose)

        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_summary.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, indent=2)
            return output_path
        return {"summary": summary}

    def _generate_cot() -> Any:
        from datacreek.generators.cot_generator import COTGenerator

        generator = COTGenerator(client, config_path, config_overrides=config_overrides)

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_examples = generation_config.num_cot_examples
        else:
            num_examples = num_pairs

        cot_result = generator.process_document(
            document_text,
            num_examples=num_examples,
            include_simple_steps=verbose,
        )

        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_cot_examples.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cot_result.to_dict(), f, indent=2)
            if verbose and cot_result.cot_examples:
                first_example = cot_result.cot_examples[0]
                logger.debug(
                    "First CoT Example:\nQuestion: %s\nReasoning: %s...\nAnswer: %s",
                    first_example.question,
                    first_example.reasoning[:100],
                    first_example.answer,
                )
            return output_path
        return cot_result.to_dict()

    def _cot_enhance() -> Any:
        from tqdm import tqdm

        from datacreek.generators.cot_generator import COTGenerator

        # Initialize the CoT generator
        generator = COTGenerator(client, config_path, config_overrides=config_overrides)

        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        # Get max_examples from args or config
        max_examples = None
        if num_pairs is not None:
            max_examples = num_pairs  # If user specified a number, use it
        else:
            config = client.config
            generation_config = get_generation_config(config)
            # Get the config value (will be None by default, meaning enhance all)
            max_examples = generation_config.num_cot_enhance_examples

        # Instead of parsing as text, load the file as JSON with conversations
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file."
            )

        # Handle different dataset formats
        # First, check for QA pairs format (the most common input format)
        if isinstance(data, dict) and "qa_pairs" in data:
            from datacreek.utils.llm_processing import convert_to_conversation_format

            qa_pairs = data.get("qa_pairs", [])
            if verbose:
                logger.debug("Converting %d QA pairs to conversation format", len(qa_pairs))

            conv_list = convert_to_conversation_format(qa_pairs)
            conversations = [{"conversations": conv} for conv in conv_list]
            is_single_conversation = False
        elif isinstance(data, dict) and "conversations" in data:
            conversations = [data]
            is_single_conversation = True
        elif isinstance(data, list) and all(
            "conversations" in item for item in data if isinstance(item, dict)
        ):
            conversations = data
            is_single_conversation = False
        elif isinstance(data, list) and all(
            isinstance(msg, dict) and "from" in msg for msg in data
        ):
            conversations = [{"conversations": data}]
            is_single_conversation = True
        else:
            conversations = data
            is_single_conversation = False

            # Limit the number of conversations if needed
            if max_examples is not None and len(conversations) > max_examples:
                if verbose:
                    logger.debug(
                        "Limiting to %d conversations (from %d total)",
                        max_examples,
                        len(conversations),
                    )
                conversations = conversations[:max_examples]

            if verbose:
                logger.debug("Found %d conversation(s) to enhance", len(conversations))

            # Process each conversation
            enhanced_conversations = []

            for i, conversation in enumerate(tqdm(conversations, desc="Enhancing conversations")):
                # Check if this item has a conversations field
                if isinstance(conversation, dict) and "conversations" in conversation:
                    conv_messages = conversation["conversations"]

                    # Validate messages format
                    if not isinstance(conv_messages, list):
                        logger.warning("conversations field is not a list in item %d, skipping", i)
                        enhanced_conversations.append(conversation)  # Keep original
                        continue

                    # Enhance this conversation's messages
                    if verbose:
                        logger.debug("Conv_messages type: %s", type(conv_messages))
                        logger.debug(
                            "Conv_messages structure: %s",
                            conv_messages[:1] if isinstance(conv_messages, list) else "Not a list",
                        )

                    # Always include simple steps when enhancing QA pairs
                    enhanced_messages = generator.enhance_with_cot(
                        conv_messages, include_simple_steps=True
                    )

                    # Handle nested bug
                    if enhanced_messages and isinstance(enhanced_messages, list):
                        # Nested bug
                        if enhanced_messages and isinstance(enhanced_messages[0], list):
                            if verbose:
                                logger.debug("Flattening nested array response")
                            enhanced_messages = enhanced_messages[0]

                    # Create enhanced conversation with same structure
                    enhanced_conv = conversation.copy()
                    enhanced_conv["conversations"] = enhanced_messages
                    enhanced_conversations.append(enhanced_conv)
                else:
                    # Not the expected format, just keep original
                    enhanced_conversations.append(conversation)

        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.json")
            with open(output_path, "w", encoding="utf-8") as f:
                if is_single_conversation and len(enhanced_conversations) == 1:
                    json.dump(enhanced_conversations[0], f, indent=2)
                else:
                    json.dump(enhanced_conversations, f, indent=2)
            if verbose:
                logger.debug("Enhanced %d conversation(s)", len(enhanced_conversations))
            return output_path

        if is_single_conversation and len(enhanced_conversations) == 1:
            return enhanced_conversations[0]
        return enhanced_conversations

    def _vqa_reasoning() -> Any:
        generator = VQAGenerator(client, config_path, config_overrides=config_overrides)

        return generator.process_dataset(
            dataset_source=file_path,
            output_dir=output_dir if save_to_file else None,
            num_examples=num_pairs,
            verbose=verbose,
        )

    def _generate_from_kg() -> Any:
        from datacreek.core.knowledge_graph import KnowledgeGraph
        from datacreek.generators.kg_generator import KGGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        kg = KnowledgeGraph()
        kg.add_document("doc", source=file_path or "inline", text=document_text)

        generator = KGGenerator(client, config_path, config_overrides=config_overrides)

        result = generator.process_graph(
            kg,
            num_pairs=num_pairs or get_generation_config(client.config).num_pairs,
            verbose=verbose,
        )

        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_kg_pairs.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    def _tool_call() -> Any:
        from datacreek.generators.tool_generator import ToolCallGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        generator = ToolCallGenerator(client, config_path, config_overrides=config_overrides)
        result = generator.process_document(document_text, verbose=verbose)
        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_tool.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    def _conversation() -> Any:
        from datacreek.generators.conversation_generator import ConversationGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        generator = ConversationGenerator(client, config_path, config_overrides=config_overrides)
        result = generator.process_document(document_text, verbose=verbose)
        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_conversation.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    def _multi_tool() -> Any:
        from datacreek.generators.multi_tool_generator import MultiToolGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        generator = MultiToolGenerator(client, config_path, config_overrides=config_overrides)
        result = generator.process_document(document_text, verbose=verbose)
        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_multi_tool.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    def _pref_pair() -> Any:
        from datacreek.generators.pref_generator import PrefPairGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        generator = PrefPairGenerator(client, config_path, config_overrides=config_overrides)
        result = generator.process_document(document_text, verbose=verbose)
        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_pref_pair.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    def _pref_list() -> Any:
        from datacreek.generators.pref_generator import PrefListGenerator

        nonlocal document_text
        if document_text is None and file_path:
            document_text = load_document_text(file_path)

        generator = PrefListGenerator(client, config_path, config_overrides=config_overrides)
        result = generator.process_document(document_text, verbose=verbose)
        if save_to_file:
            output_path = os.path.join(output_dir, f"{base_name}_pref_list.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return output_path
        return result

    handlers = {
        ContentType.QA: _generate_qa,
        ContentType.SUMMARY: _generate_summary,
        ContentType.COT: _generate_cot,
        ContentType.COT_ENHANCE: _cot_enhance,
        ContentType.VQA_ADD_REASONING: _vqa_reasoning,
        ContentType.FROM_KG: _generate_from_kg,
        ContentType.TOOL_CALL: _tool_call,
        ContentType.CONVERSATION: _conversation,
        ContentType.MULTI_TOOL: _multi_tool,
        ContentType.PREF_PAIR: _pref_pair,
        ContentType.PREF_LIST: _pref_list,
    }

    if ct not in handlers:
        raise ValueError(f"Unknown content type: {content_type}")

    return handlers[ct]()
