# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Filter low quality examples

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_curate_config, get_prompt
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format, parse_ratings

logger = logging.getLogger(__name__)


def curate_qa_pairs(
    input_path: str,
    output_path: str,
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    batch_size: Optional[int] = None,
    inference_batch: Optional[int] = None,
) -> str:
    """Clean and filter QA pairs based on quality ratings

    Args:
        input_path: Path to the input file with QA pairs
        output_path: Path to save the cleaned output
        threshold: Quality threshold (1-10)
        api_base: VLLM API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed output
        batch_size: Override batch size for rating
        inference_batch: Number of batches to send at once

    Returns:
        Path to the cleaned output file
    """
    # Set verbose either via CLI or via env variable. If its via CLI, set it to env variable
    if verbose:
        os.environ["SDK_VERBOSE"] = "true"
    else:
        os.environ["SDK_VERBOSE"] = "false"

    # Load input file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract QA pairs
    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")

    # If there are no QA pairs or they're already filtered
    if not qa_pairs:
        raise ValueError("No QA pairs found in the input file")

    # Initialize LLM client
    client = LLMClient(
        config_path=config_path, provider=provider, api_base=api_base, model_name=model
    )

    # Get threshold from args, then config, then default
    if threshold is None:
        config = client.config
        cleanup_config = get_curate_config(config)
        threshold = cleanup_config.get("threshold", 7.0)

    # Create QA generator
    generator = QAGenerator(client, config_path)

    # Get configuration
    curate_config = get_curate_config(client.config)

    if batch_size is None:
        batch_size = curate_config.get("batch_size", 32)
    if inference_batch is None:
        inference_batch = curate_config.get("inference_batch", 32)

    rating_temperature = curate_config.get("temperature", 0.1)

    if threshold is None:
        threshold = curate_config.get("threshold", 7.0)

    # Get rating prompt template
    rating_prompt_template = get_prompt(client.config, "qa_rating")

    # Split QA pairs into batches
    batches = []
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i : i + batch_size]
        batches.append(batch)

    # Prepare all message batches for rating
    all_messages = []
    for batch in batches:
        batch_json = json.dumps(batch, indent=2)
        rating_prompt = rating_prompt_template.format(pairs=batch_json)
        messages = [{"role": "system", "content": rating_prompt}]
        all_messages.append(messages)

    # Initialize counters and result containers
    filtered_pairs = []
    total_score = 0
    total_evaluated = 0
    total_passed = 0

    # Process batches with simple progress indicator rather than a detailed bar
    # This avoids conflicts with other output messages
    logger.info("Processing %d batches of QA pairs...", len(batches))

    # Only use detailed progress bar in verbose mode
    if verbose:
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        progress_ctx = Progress(*progress_columns)
        rate_task = progress_ctx.add_task(f"Rating QA pairs", total=len(batches))
        progress_ctx.start()
    else:
        progress_ctx = None
        rate_task = None

    # Process in inference batches
    for batch_start in range(0, len(all_messages), inference_batch):
        batch_end = min(batch_start + inference_batch, len(all_messages))
        current_batch = all_messages[batch_start:batch_end]
        current_batch_size = len(current_batch)

        batch_num = batch_start // inference_batch + 1
        total_batches = (len(all_messages) + inference_batch - 1) // inference_batch

        # Simple progress indicator for non-verbose mode
        if not verbose:
            logger.info("Processing batch %d/%d...", batch_num, total_batches)
        else:
            logger.info("Processing batch %d/%d", batch_num, total_batches)

        try:
            # Get ratings for the batch
            if verbose:
                logger.debug("Sending batch request with %d items", len(current_batch))

            batch_responses = client.batch_completion(
                current_batch, temperature=rating_temperature, batch_size=inference_batch
            )

            if verbose:
                logger.debug("Received %d responses", len(batch_responses))
                for i, resp in enumerate(batch_responses):
                    logger.debug("Response %d: %s", i + 1, resp[:100])

            # Process each response
            for j, response in enumerate(batch_responses):
                original_batch_index = batch_start + j
                if original_batch_index < len(batches):
                    original_batch = batches[original_batch_index]

                    # Parse the ratings with original batch for fallback
                    try:
                        if verbose:
                            logger.debug("Processing batch %d", original_batch_index + 1)

                        rated_batch = parse_ratings(response, original_batch)

                        # Process the rated batch
                        for pair in rated_batch:
                            if "rating" in pair:
                                rating = pair["rating"]
                                total_score += rating
                                total_evaluated += 1

                                if rating >= threshold:
                                    filtered_pairs.append(pair)
                                    total_passed += 1
                    except Exception as e:
                        if verbose:
                            logger.error(
                                "Error processing batch %d: %s", original_batch_index + 1, str(e)
                            )
                            logger.debug("First 100 chars of response: %s", response[:100])

                        # Try processing one pair at a time as a fallback
                        try:
                            if verbose:
                                logger.debug("Attempting to process items individually...")

                            for item in original_batch:
                                item_json = json.dumps(item, indent=2)
                                rating_prompt = rating_prompt_template.format(pairs=item_json)
                                item_response = client.chat_completion(
                                    [{"role": "system", "content": rating_prompt}],
                                    temperature=rating_temperature,
                                )
                                try:
                                    # This should be a single item
                                    rated_item = parse_ratings(item_response, [item])
                                    if rated_item and len(rated_item) > 0:
                                        pair = rated_item[0]
                                        if "rating" in pair:
                                            rating = pair["rating"]
                                            total_score += rating
                                            total_evaluated += 1

                                            if rating >= threshold:
                                                filtered_pairs.append(pair)
                                                total_passed += 1
                                                if verbose:
                                                    logger.debug(
                                                        "Successfully processed individual item with rating %s",
                                                        rating,
                                                    )
                                except Exception as inner_e:
                                    if verbose:
                                        logger.error(
                                            "Failed to process individual item: %s", str(inner_e)
                                        )
                        except Exception as fallback_e:
                            if verbose:
                                logger.error("Fallback processing failed: %s", str(fallback_e))

                        # Continue processing other batches rather than failing completely
                        pass

            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)

        except Exception as e:
            if verbose:
                logger.error("Error processing inference batch %d: %s", batch_num, str(e))

            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)

    # Stop progress bar if in verbose mode
    if progress_ctx:
        progress_ctx.stop()

    # Clear the progress line in non-verbose mode
    if not verbose:
        logger.info("Batch processing complete.")

    # Calculate metrics
    metrics = {
        "total": len(qa_pairs),
        "filtered": len(filtered_pairs),
        "retention_rate": round(len(filtered_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
        "avg_score": round(total_score / total_evaluated, 1) if total_evaluated else 0,
    }

    # Always print basic stats, even in non-verbose mode
    logger.info("Rated %d QA pairs", total_evaluated)
    logger.info("Retained %d pairs (threshold: %s)", total_passed, threshold)
    logger.info("Average score: %s", metrics["avg_score"])

    # Convert to conversation format
    conversations = convert_to_conversation_format(filtered_pairs)

    # Create result with filtered pairs
    result = {
        "summary": summary,
        "qa_pairs": filtered_pairs,
        "conversations": conversations,
        "metrics": metrics,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return output_path
