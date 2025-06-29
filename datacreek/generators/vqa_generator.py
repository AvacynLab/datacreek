# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Visual Question Answering Generator

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_generation_config, load_config

logger = logging.getLogger(__name__)

# Note: The following packages are required for this module:
# - openai: For API access to vision models
# - datasets: For handling HuggingFace datasets
# - huggingface_hub: For accessing HuggingFace repositories


def _check_optional_deps() -> None:
    """Ensure optional dependencies are available."""
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime import
        raise ImportError(
            "The 'datasets' and 'huggingface_hub' packages are required for VQA generation."
        ) from exc


class VQAGenerator:
    """Generates Visual Question Answering data with reasoning"""

    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the VQA Generator with an LLM client and optional config"""

        _check_optional_deps()

        self.client = client

        if config_path or config_overrides:
            from datacreek.utils.config import load_config_with_overrides

            self.config = load_config_with_overrides(
                str(config_path) if config_path else None, config_overrides
            )
        else:
            self.config = client.config

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)

    def encode_image_base64(self, image):
        """Encode an image in base64 format"""
        import base64
        import io

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def transform(self, messages):
        """Transform messages by adding reasoning to VQA data"""
        verbose = logger.isEnabledFor(logging.DEBUG)

        # Get prompt from config
        prompt = self.config.get("prompt", "")

        # Get generation config
        temperature = self.generation_config.temperature
        max_tokens = self.generation_config.max_tokens
        batch_size = self.generation_config.batch_size

        # Process the messages from the dataset
        # Create a list of message sets for the model
        messages_list = []

        for i in range(len(messages["image"])):
            image = messages["image"][i]
            query = messages["query"][i]
            label = (
                messages["label"][i][0]
                if isinstance(messages["label"][i], list)
                else messages["label"][i]
            )

            # Encode the image
            image_base64 = self.encode_image_base64(image)

            # Prepare the messages for the API request
            message_set = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {"type": "text", "text": f"{query} Final answer: {label}"},
                    ],
                },
            ]
            messages_list.append(message_set)

        if verbose:
            logger.info("Processing %d VQA items...", len(messages_list))

        # Use the client's batch_completion method instead of our own async implementation
        results = self.client.batch_completion(
            message_batches=messages_list,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )

        for i, response in enumerate(results):
            # Update the messages with the response
            messages["label"][i] = response

            if verbose and i < 2:  # Show first two examples in verbose mode
                logger.info("Example %d:", i + 1)
                logger.info("Query: %s", messages["query"][i])
                logger.info(
                    "Response: %s",
                    response[:100] + "..." if len(response) > 100 else response,
                )

        return messages

    def process_dataset(
        self,
        dataset_source,
        output_dir: str,
        num_examples: Optional[int] = None,
        input_split: Optional[str] = None,
        output_split: Optional[str] = None,
        verbose: bool = False,
    ) -> str:
        """Process a dataset to add reasoning to VQA data

        Args:
            dataset_source: Dataset source (path or HuggingFace dataset ID)
            output_dir: Directory to save the processed dataset
            num_examples: Maximum number of examples to process
            input_split: Dataset split to use as input
            output_split: Dataset split to use for output
            verbose: Whether to print verbose output

        Returns:
            Path to the output dataset
        """
        verbose = logger.isEnabledFor(logging.DEBUG)

        try:
            # Try to load from file
            try:
                from datasets import Dataset
            except ImportError:
                raise ImportError(
                    "The 'datasets' package is required for this functionality. Please install it using 'pip install datasets'."
                )
            try:
                with open(dataset_source, "r", encoding="utf-8") as f:
                    input_data = f.read()
                dataset = Dataset.from_dict(json.loads(input_data))
            except FileNotFoundError as e:
                # If the file doesn't exist, try to load it from the dataset hub
                try:
                    from datasets import load_dataset
                    from huggingface_hub import HfApi
                except ImportError:
                    raise ImportError(
                        "The 'huggingface_hub' and 'datasets' packages are required for this functionality. Please install them using 'pip install huggingface_hub datasets'."
                    )

                hf_api = HfApi()
                if hf_api.repo_exists(repo_id=dataset_source, repo_type="dataset"):
                    dataset = load_dataset(dataset_source)
                else:
                    # Uplevel error
                    raise e

            # Get input split from config if not provided
            if input_split is None:
                input_split = self.config.get("input_split", None)

            # Get output split from config if not provided
            if output_split is None:
                output_split = self.config.get("output_split", None)

            # Use the specified split if provided
            if input_split is not None:
                dataset = dataset[input_split]

            # Get max_examples from args or config
            max_examples = num_examples
            if max_examples is not None and max_examples > 0:
                # Limit the dataset size
                dataset = dataset.select(range(min(max_examples, len(dataset))))

            if verbose:
                logger.info("Processing %d examples from dataset", len(dataset))

            # Get batch size from config
            batch_size = self.generation_config.batch_size

            if verbose:
                logger.info("Using batch size of %d for dataset processing", batch_size)

            # Process the dataset
            ds = dataset.map(
                self.transform,
                batch_size=batch_size,
                batched=True,
            )

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save the processed dataset
            if output_split is not None:
                # Create the split directory
                os.makedirs(f"{output_dir}/{output_split}", exist_ok=True)

                # Write output that can be loaded back in with load_dataset
                ds.to_parquet(f"{output_dir}/{output_split}/data.parquet")
                meta_data = {"splits": [output_split]}
                with open(f"{output_dir}/dataset_dict.json", "w") as f:
                    f.write(json.dumps(meta_data, indent=4))

                output_path = f"{output_dir}/{output_split}/data.parquet"
            else:
                # Just dump it in a parquet file
                ds.to_parquet(f"{output_dir}/data.parquet")
                output_path = f"{output_dir}/data.parquet"

            if verbose:
                logger.info("Saved processed dataset to %s", output_path)

            return output_path

        except Exception as e:
            logger.error("Error processing dataset: %s", e)
            raise
