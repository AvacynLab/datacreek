# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.generators.vqa_generator import VQAGenerator
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_generation_config

logger = logging.getLogger(__name__)


def read_json(file_path):
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    return document_text


def _generate_qa(
    base_name: str,
    file_path: str,
    output_dir: str,
    client: LLMClient,
    config_path: Optional[Path],
    num_pairs: Optional[int],
    verbose: bool,
) -> str:
    generator = QAGenerator(client, config_path)

    document_text = read_json(file_path)

    if num_pairs is None:
        generation_config = get_generation_config(client.config)
        num_pairs = generation_config.get("num_pairs", 25)

    result = generator.process_document(document_text, num_pairs=num_pairs, verbose=verbose)

    output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    logger.info("Saving result to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return output_path


def _generate_summary(
    base_name: str,
    file_path: str,
    output_dir: str,
    client: LLMClient,
    config_path: Optional[Path],
    num_pairs: Optional[int],
    verbose: bool,
) -> str:
    generator = QAGenerator(client, config_path)
    document_text = read_json(file_path)
    summary = generator.generate_summary(document_text)
    output_path = os.path.join(output_dir, f"{base_name}_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=2)
    return output_path


def _generate_cot(
    base_name: str,
    file_path: str,
    output_dir: str,
    client: LLMClient,
    config_path: Optional[Path],
    num_pairs: Optional[int],
    verbose: bool,
) -> str:
    from synthetic_data_kit.generators.cot_generator import COTGenerator

    generator = COTGenerator(client, config_path)

    document_text = read_json(file_path)

    if num_pairs is None:
        generation_config = get_generation_config(client.config)
        num_pairs = generation_config.get("num_cot_examples", 5)

    result = generator.process_document(
        document_text,
        num_examples=num_pairs,
        include_simple_steps=verbose,
    )

    output_path = os.path.join(output_dir, f"{base_name}_cot_examples.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose and result.get("cot_examples"):
        logger.debug("First CoT Example: %s", result["cot_examples"][0])

    return output_path


def _enhance_cot(
    base_name: str,
    file_path: str,
    output_dir: str,
    client: LLMClient,
    config_path: Optional[Path],
    num_pairs: Optional[int],
    verbose: bool,
) -> str:
    from synthetic_data_kit.generators.cot_generator import COTGenerator
    from tqdm import tqdm

    generator = COTGenerator(client, config_path)

    max_examples = None
    if num_pairs is not None:
        max_examples = num_pairs
    else:
        generation_config = get_generation_config(client.config)
        max_examples = generation_config.get("num_cot_enhance_examples")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "qa_pairs" in data:
            from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format

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

        if max_examples is not None and len(conversations) > max_examples:
            if verbose:
                logger.debug(
                    "Limiting to %d conversations (from %d total)",
                    max_examples,
                    len(conversations),
                )
            conversations = conversations[:max_examples]

        if verbose:
            logger.info("Found %d conversation(s) to enhance", len(conversations))

        enhanced_conversations = []
        for i, conversation in enumerate(tqdm(conversations, desc="Enhancing conversations")):
            if isinstance(conversation, dict) and "conversations" in conversation:
                conv_messages = conversation["conversations"]
                if not isinstance(conv_messages, list):
                    logger.warning("conversations field is not a list in item %d, skipping", i)
                    enhanced_conversations.append(conversation)
                    continue

                if verbose:
                    logger.debug("Conv_messages type: %s", type(conv_messages))
                    logger.debug(
                        "Conv_messages structure: %s",
                        conv_messages[:1] if isinstance(conv_messages, list) else "Not a list",
                    )

                enhanced_messages = generator.enhance_with_cot(
                    conv_messages, include_simple_steps=True
                )

                if enhanced_messages and isinstance(enhanced_messages, list):
                    if enhanced_messages and isinstance(enhanced_messages[0], list):
                        if verbose:
                            logger.debug("Flattening nested array response")
                        enhanced_messages = enhanced_messages[0]

                enhanced_conv = conversation.copy()
                enhanced_conv["conversations"] = enhanced_messages
                enhanced_conversations.append(enhanced_conv)
            else:
                enhanced_conversations.append(conversation)

        output_path = os.path.join(output_dir, f"{base_name}_enhanced.json")
        with open(output_path, "w", encoding="utf-8") as f:
            if is_single_conversation and len(enhanced_conversations) == 1:
                json.dump(enhanced_conversations[0], f, indent=2)
            else:
                json.dump(enhanced_conversations, f, indent=2)

        if verbose:
            logger.info("Enhanced %d conversation(s)", len(enhanced_conversations))

        return output_path
    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file."
        )


def _vqa_add_reasoning(
    base_name: str,
    file_path: str,
    output_dir: str,
    client: LLMClient,
    config_path: Optional[Path],
    num_pairs: Optional[int],
    verbose: bool,
) -> str:
    generator = VQAGenerator(client, config_path)
    output_path = generator.process_dataset(
        dataset_source=file_path,
        output_dir=output_dir,
        num_examples=num_pairs,
        verbose=verbose,
    )
    return output_path


def process_file(
    file_path: str,
    output_dir: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
) -> str:
    """Process a file to generate content

    Args:
        file_path: Path to the text file to process
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate (qa, summary, cot)
        num_pairs: Target number of QA pairs to generate
        threshold: Quality threshold for filtering (1-10)

    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    # The reason for having this directory logic for now is explained in context.py
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LLM client
    client = LLMClient(
        config_path=config_path, provider=provider, api_base=api_base, model_name=model
    )

    logger.info("Using %s provider", client.provider)

    # Generate base filename for output
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Dispatch to handler based on content type
    handlers = {
        "qa": _generate_qa,
        "summary": _generate_summary,
        "cot": _generate_cot,
        "cot-enhance": _enhance_cot,
        "vqa_add_reasoning": _vqa_add_reasoning,
    }

    if content_type not in handlers:
        raise ValueError(f"Unknown content type: {content_type}")

    handler = handlers[content_type]
    return handler(base_name, file_path, output_dir, client, config_path, num_pairs, verbose)
