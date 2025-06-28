# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Filter low quality examples

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.generators.qa_generator import QAGenerator
from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_curate_settings, get_prompt

logger = logging.getLogger(__name__)
from datacreek.utils.llm_processing import convert_to_conversation_format, parse_ratings


def curate_qa_pairs(
    input_data: Any,
    output_path: Optional[str] = None,
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    async_mode: bool = False,
) -> Any:
    """Clean and filter QA pairs based on quality ratings

    Args:
        input_path: Path to the input file with QA pairs
        output_path: Path to save the cleaned output
        threshold: Quality threshold (1-10)
        api_base: VLLM API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed output
        async_mode: Use asynchronous LLM requests when supported

    Returns:
        Path to the cleaned output file
    """
    # Verbosity now controlled by logging level

    # Load input
    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(input_data)
    else:
        data = input_data

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
        cleanup_settings = get_curate_settings(config)
        threshold = cleanup_settings.threshold

    # Create QA generator
    generator = QAGenerator(client, config_path)

    # Get configuration
    curate_config = get_curate_settings(client.config)

    batch_size = curate_config.batch_size
    inference_batch = curate_config.inference_batch

    rating_temperature = curate_config.temperature

    if threshold is None:
        threshold = curate_config.threshold

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

    from datacreek.utils.batch import async_process_batches, process_batches

    def _parse_and_collect(resp: str, original_batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        rated = parse_ratings(resp, original_batch)
        collected: List[Dict[str, Any]] = []
        for pair in rated:
            if "rating" in pair:
                rating = pair["rating"]
                nonlocal total_score, total_evaluated, total_passed
                total_score += rating
                total_evaluated += 1
                if rating >= threshold:
                    collected.append(pair)
                    total_passed += 1
        return collected

    if async_mode:
        rated_batches = asyncio.run(
            async_process_batches(
                client,
                all_messages,
                batch_size=inference_batch,
                temperature=rating_temperature,
                parse_fn=lambda resp: resp,
            )
        )
    else:
        rated_batches = process_batches(
            client,
            all_messages,
            batch_size=inference_batch,
            temperature=rating_temperature,
            parse_fn=lambda resp: resp,
        )

    for idx, response in enumerate(rated_batches):
        original_batch = batches[idx] if idx < len(batches) else []
        try:
            filtered_pairs.extend(_parse_and_collect(response, original_batch))
        except Exception as e:
            logger.error("Error processing batch %d: %s", idx + 1, e)

        if progress_ctx and rate_task:
            progress_ctx.update(rate_task, advance=1)

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

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return output_path

    return result
