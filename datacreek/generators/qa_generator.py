# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_curate_config, get_generation_config, get_prompt, load_config
from datacreek.utils.llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
)
from datacreek.utils.text import split_into_chunks

logger = logging.getLogger(__name__)


class QAGenerator:
    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        kg: Optional[KnowledgeGraph] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client
        self.kg = kg

        # Load config and merge overrides if provided
        self.config = load_config(config_path)
        if config_overrides:
            from datacreek.utils.config import merge_configs

            self.config = merge_configs(self.config, config_overrides)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)

    def generate_summary(self, document_text: str) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"
        if verbose:
            logger.info("Generating document summary...")

        # Get summary prompt from config
        prompt = get_prompt(self.config, "summary")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": document_text},
        ]

        temperature = self.generation_config.summary_temperature
        max_tokens = self.generation_config.summary_max_tokens
        summary = self.client.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        )

        if verbose:
            logger.info("Summary generated (%d chars)", len(summary))
        return summary

    def generate_qa_pairs(
        self,
        document_text: str,
        summary: str,
        num_pairs: int = 25,
        query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Generate QA pairs from the document using batched processing"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

        # Get generation config
        chunk_size = self.generation_config.chunk_size
        temperature = self.generation_config.temperature
        overlap = self.generation_config.overlap
        batch_size = self.generation_config.batch_size
        chunk_method = self.generation_config.chunk_method
        similarity_drop = self.generation_config.similarity_drop
        top_k = self.generation_config.retrieval_top_k

        # Split text into chunks
        chunks = split_into_chunks(
            document_text,
            chunk_size=chunk_size,
            overlap=overlap,
            method=chunk_method,
            similarity_drop=similarity_drop,
        )

        if self.kg and chunk_method in {"sliding", "semantic", "contextual"}:
            # index chunks in knowledge graph if provided
            for i, chunk in enumerate(chunks):
                cid = f"chunk-{i}"
                self.kg.add_document("doc", source="inline") if "doc" not in self.kg.graph else None
                self.kg.add_chunk("doc", cid, chunk)

        if query and self.kg:
            selected_ids = self.kg.search_embeddings(query, k=top_k)
            chunks = [self.kg.graph.nodes[c]["text"] for c in selected_ids if c in self.kg.graph]

        if verbose:
            logger.info("Generating QA pairs...")
            logger.info("Document split into %d chunks", len(chunks))
            logger.info("Using batch size of %d", batch_size)

        all_qa_pairs = []
        pairs_per_chunk = max(1, round(num_pairs / len(chunks)))

        # Get QA generation prompt template
        qa_prompt_template = get_prompt(self.config, "qa_generation")

        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with summary and text
            qa_prompt = qa_prompt_template.format(
                num_pairs=pairs_per_chunk, summary=summary[:100], text=chunk
            )

            messages = [{"role": "system", "content": qa_prompt}]
            all_messages.append(messages)

        logger.info("Processing %d chunks to generate QA pairs...", len(chunks))

        # Set up progress tracking based on verbose mode
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
            generate_task = progress_ctx.add_task(f"Generating QA pairs", total=len(chunks))
            progress_ctx.start()
        else:
            progress_ctx = None
            generate_task = None

        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_messages = all_messages[batch_start:batch_end]
            current_batch_size = len(batch_messages)

            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            # Simple progress indicator for non-verbose mode
            if not verbose:
                logger.info(
                    "Processing batch %d/%d...",
                    batch_num,
                    total_batches,
                )
            else:
                logger.info(
                    "Processing batch %d/%d with %d chunks",
                    batch_num,
                    total_batches,
                    current_batch_size,
                )

            try:
                # Process the batch
                batch_responses = self.client.batch_completion(
                    batch_messages,
                    temperature=temperature,
                    batch_size=batch_size,
                )

                # Process each response in the batch
                for j, response in enumerate(batch_responses):
                    chunk_index = batch_start + j
                    chunk_pairs = parse_qa_pairs(response)
                    all_qa_pairs.extend(chunk_pairs)

                    if verbose:
                        logger.info(
                            "  Generated %d pairs from chunk %d",
                            len(chunk_pairs),
                            chunk_index + 1,
                        )

                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

            except Exception as e:
                if verbose:
                    logger.error("  Error processing batch %d: %s", batch_num, str(e))

                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

        # Stop progress bar if in verbose mode
        if progress_ctx:
            progress_ctx.stop()

        # Clear the progress line in non-verbose mode
        if not verbose:
            logger.info("Batch processing complete.")

        logger.info("Generated %d QA pairs total", len(all_qa_pairs))
        return all_qa_pairs

    def rate_qa_pairs(
        self, qa_pairs: List[Dict[str, str]], summary: str, threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rate and filter QA pairs by quality"""
        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}

        # Get threshold from args, then config, then default
        if threshold is None:
            threshold = self.curate_config.get("threshold", 7.0)

        if verbose:
            logger.info("Evaluating %d pairs...", len(qa_pairs))

        # Get rating config
        batch_size = self.curate_config.get("batch_size", 8)
        temperature = self.curate_config.get("temperature", 0.1)

        # Get rating prompt template
        rating_prompt_template = get_prompt(self.config, "qa_rating")

        # Process in batches
        batches = [qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)]

        rated_pairs = []
        total_score = 0

        # Create progress bar
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns) as progress:
            rating_task = progress.add_task(f"Rating QA pairs", total=len(batches))

            for i, batch in enumerate(batches):
                if verbose:
                    logger.info("Rating batch %d/%d...", i + 1, len(batches))
                batch_json = json.dumps(batch, indent=2)

                # Format the rating prompt with pairs
                rating_prompt = rating_prompt_template.format(pairs=batch_json)

                messages = [{"role": "system", "content": rating_prompt}]

                try:
                    response = self.client.chat_completion(messages, temperature=temperature)

                    rated_batch = parse_ratings(response)

                    for pair in rated_batch:
                        if "rating" in pair:
                            total_score += pair["rating"]
                            if pair["rating"] >= threshold:
                                rated_pairs.append(pair)

                except Exception as e:
                    if verbose:
                        logger.error("Error rating batch %d: %s", i + 1, str(e))

                time.sleep(0.5)  # Avoid rate limits
                progress.update(rating_task, advance=1)

        # Calculate metrics
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0,
        }

        # Always print summary information, even in non-verbose mode
        logger.info(
            "Keeping %d out of %d pairs (threshold: %s)",
            len(rated_pairs),
            len(qa_pairs),
            threshold,
        )
        logger.info("Average score: %s", metrics["avg_score"])
        return rated_pairs, metrics

    def process_document(
        self, document_text: str, num_pairs: int = 25, verbose: bool = False
    ) -> Dict[str, Any]:
        """Process a document to generate QA pairs without rating"""
        # Set the verbose environment variable
        if verbose:
            os.environ["SDK_VERBOSE"] = "true"
        else:
            os.environ["SDK_VERBOSE"] = "false"

        # Generate summary
        summary = self.generate_summary(document_text)

        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(document_text, summary, num_pairs=num_pairs)

        # Prepare result - no rating at this stage
        result = {"summary": summary, "qa_pairs": qa_pairs}

        return result
